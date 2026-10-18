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
