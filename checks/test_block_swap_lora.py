"""Block swap with a LoRA on some blocks only. MOSS-Image leaves block 22 without a LoRA (the negative branch skips
it), so when its SFT preview checkpoint streams blocks with the epoch's LoRA on, the swap pairs an unwrapped block
with a wrapped one. Pairing by raw module name met the LoRALinear itself ("'LoRALinear' object has no attribute
'weight'") one way round, and left the unwrapped block's weights on the GPU the other way round."""
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from fizgig.families import registry  # noqa: E402
from fizgig.families.lora import FamilyLoRA, LoRALinear  # noqa: E402
from fizgig.modules import offloading  # noqa: E402

TINY = dict(dim=32, ffn_dim=48, num_heads=4, num_layers=24, text_dim=16, freq_dim=16, latent_channels=8)


class _Reached(Exception):
    """The weight-pairing step finished; the CUDA copies come next (no GPU on the check machine)."""


@pytest.fixture(scope="module")
def wrapped():
    from fizgig.moss_image.model import MOSSImageDiT
    drv = registry.get("moss_image").load_driver()
    dit = MOSSImageDiT(**TINY).eval().requires_grad_(False)
    net = FamilyLoRA(dit, drv, device="cpu")
    blocks = list(dit.layers)
    assert isinstance(blocks[21].proj["q"], LoRALinear)
    assert type(blocks[22].proj["q"]) is nn.Linear           # block 22: no LoRA (UNCOND_SKIP)
    return blocks


def _linears(block):
    return {id(m) for m in block.modules() if type(m) is nn.Linear}   # the base weights a swap must move


@pytest.mark.parametrize("to_cpu,to_cuda", [(20, 22), (22, 20), (20, 21)])
def test_cuda_swap_pairs_wrapped_with_unwrapped(wrapped, monkeypatch, to_cpu, to_cuda):
    class _Stream:
        def synchronize(self):
            raise _Reached
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **k: _Stream())
    off = offloading.Offloader("moss_image", len(wrapped), 20, torch.device("cpu"))
    with pytest.raises(_Reached):
        off.swap_weight_devices_cuda(torch.device("cpu"), wrapped[to_cpu], wrapped[to_cuda])


@pytest.mark.parametrize("to_cpu,to_cuda", [(20, 22), (22, 20), (20, 21)])
def test_every_base_weight_has_a_partner(wrapped, to_cpu, to_cuda):
    """Each Linear of the block coming in is paired with the same Linear of the block going out, so both move."""
    pairs = offloading._weight_swap_pairs(wrapped[to_cpu], wrapped[to_cuda])
    assert all(c is not None for c, _ in pairs)
    assert {id(c) for c, _ in pairs} == _linears(wrapped[to_cpu])
    assert {id(m) for _, m in pairs} == _linears(wrapped[to_cuda])
    for c, m in pairs:
        assert c.weight.shape == m.weight.shape


def test_no_cuda_swap_pairs_by_position(wrapped):
    pairs = offloading._weight_swap_pairs(wrapped[21], wrapped[22], linear_only=False)
    names_cpu = {id(m): n for n, m in wrapped[21].named_modules()}
    names_cuda = {id(m): n for n, m in wrapped[22].named_modules()}
    for c, m in pairs:
        assert c is not None
        assert names_cpu[id(c)].removesuffix(".base") == names_cuda[id(m)]


# ---- INT8 base: block 22 is not quantised either (quant targets = LoRA targets) ----------------------------------
class _NoCuda:
    """Stands in for the CUDA stream / event / pinned-memory calls, so the CUDA swap path runs on the CPU."""

    def __init__(self, *a, **k):
        pass

    def synchronize(self):
        pass

    def wait_event(self, e):
        pass

    def record(self, *a):
        pass

    def record_event(self):
        return _NoCuda()


@pytest.fixture
def cuda_on_cpu(monkeypatch):
    import contextlib
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **k: _NoCuda())
    monkeypatch.setattr(torch.cuda, "stream", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", _NoCuda)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self, device=None: self)


def _int8_preview_model():
    """The SFT preview model as load_preview_checkpoint builds it on a small card: INT8 base, the epoch's LoRA."""
    from fizgig.families import quant
    from fizgig.moss_image.model import MOSSImageDiT
    drv = registry.get("moss_image").load_driver()
    torch.manual_seed(0)
    dit = MOSSImageDiT(**TINY).eval().requires_grad_(False)
    quant.quantize(dit, drv, "int8", "cpu", store_device="cpu")
    net = FamilyLoRA(dit, drv, device="cpu")
    net.add_trainable(4, 4.0)
    for p in net.parameters():
        p.data.normal_(0, 0.05)
    return dit


def test_int8_and_bf16_linears_never_pair():
    dit = _int8_preview_model()
    assert dit.layers[21].proj["q"].base.weight.dtype == torch.int8
    assert dit.layers[22].proj["q"].weight.dtype != torch.int8          # block 22: no LoRA, not quantised
    for a, b in ((21, 22), (22, 21)):
        pairs, cpu_only = offloading._weight_swap_plan(dit.layers[a], dit.layers[b])
        assert all(c is None for c, _ in pairs)
        assert len(cpu_only) == len(pairs) == 6
    pairs, cpu_only = offloading._weight_swap_plan(dit.layers[20], dit.layers[21])
    assert all(c is not None for c, _ in pairs) and not cpu_only


@pytest.mark.parametrize("swap", [1, 13, 22])
def test_int8_preview_streams_like_resident(cuda_on_cpu, swap):
    """A fresh INT8 + LoRA preview model per round, as each epoch loads one: the streamed forward matches the
    resident one. 1 block keeps block 22 resident (Zaush_v4's epoch 0: 18 of 42); 13 and 22 wrap around, so block 22
    streams (epoch 1: 23 of 42 -> "expected mat1 and mat2 to have the same dtype ... BFloat16 != signed char")."""
    x = torch.randn(1, 8, 4, 4, generator=torch.Generator().manual_seed(1))
    ctx = torch.randn(1, 20, 16, generator=torch.Generator().manual_seed(2))
    t = torch.tensor([500.0])
    dit = _int8_preview_model()
    with torch.no_grad():
        ref = dit(x, t, ctx)
        dit.enable_block_swap(swap, torch.device("cpu"), supports_backward=False)
        dit.offloader.cuda_available, dit.offloader.stream = True, _NoCuda()
        dit.switch_block_swap_for_inference()
        for _ in range(2):                          # two sampling steps: the second starts from the rotated blocks
            assert torch.equal(dit(x, t, ctx), ref)
        for i in list(dit.offloader.futures):
            dit.offloader._wait_blocks_move(i)
    for n, m in dit.named_modules():
        if type(m) is nn.Linear and n.startswith("layers."):
            want = torch.int8 if not n.startswith("layers.22.") else torch.float32
            assert m.weight.dtype == want, n
