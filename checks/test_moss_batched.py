"""MOSS-Image's batched forward (batch > 1 in one pass, text padded with a key-padding mask) against the per-sample
loop on a tiny random DiT: same velocity, same LoRA gradients, and batch 1 untouched."""
import pytest

torch = pytest.importorskip("torch")
from fizgig.moss_image.model import MOSSImageDiT  # noqa: E402

TINY = dict(dim=64, ffn_dim=96, num_heads=4, num_layers=4, text_dim=48, freq_dim=32, latent_channels=16)


class _LoRA(torch.nn.Module):
    def __init__(self, base, rank=4):
        super().__init__()
        self.base = base
        self.down = torch.nn.Linear(base.in_features, rank, bias=False)
        self.up = torch.nn.Linear(rank, base.out_features, bias=False)
        torch.nn.init.normal_(self.up.weight, std=0.05)

    def forward(self, x):
        return self.base(x) + self.up(self.down(x))


def _model(dtype):
    torch.manual_seed(0)
    m = MOSSImageDiT(**TINY)
    for p in m.parameters():
        p.data.normal_(0, 0.05)
    for layer in m.layers:
        for name in ("q", "v", "ffn_down"):
            layer.proj[name] = _LoRA(layer.proj[name])
    m = m.to(dtype)
    for n, p in m.named_parameters():
        p.requires_grad_(".down." in n or ".up." in n)
    return m


def _inputs(dtype, batch=3):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(batch, 16, 4, 6, generator=g, dtype=dtype)
    ctx = torch.randn(batch, 9, 48, generator=g, dtype=dtype)
    mask = torch.zeros(batch, 9, dtype=torch.bool)
    mask[0, :9] = True
    mask[1, :4] = True
    if batch > 2:
        mask[2, [1, 3, 4, 7, 8]] = True                    # not left-aligned
    t = torch.tensor([900.0, 300.0, 50.0][:batch])
    return x, t, ctx, mask[:batch]


def _run(m, batched, x, t, ctx, mask, **kw):
    m.batched_forward = batched
    m.zero_grad(set_to_none=True)
    out = m(x, t, ctx, mask, **kw)
    (out.square().mean()).backward()
    grads = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    return out.detach(), grads


@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-10), (torch.float32, 1e-4)])
@pytest.mark.parametrize("batch", [2, 3])
@pytest.mark.parametrize("ckpt,skip", [(False, ()), (True, (2,))])
def test_batched_matches_loop(dtype, tol, batch, ckpt, skip):
    m = _model(dtype)
    m.enable_gradient_checkpointing(ckpt)
    args = _inputs(dtype, batch)
    ref, gref = _run(m, False, *args, skip_layers=skip)
    out, gout = _run(m, True, *args, skip_layers=skip)
    assert out.shape == ref.shape
    assert (out - ref).abs().max().item() < tol * max(1.0, ref.abs().max().item())
    assert gref.keys() == gout.keys()
    for n in gref:
        assert (gout[n] - gref[n]).abs().max().item() < tol * max(1.0, gref[n].abs().max().item()), n


def test_equal_lengths_need_no_mask():
    m = _model(torch.float64)
    x, t, ctx, _ = _inputs(torch.float64, 2)
    mask = torch.ones(2, 9, dtype=torch.bool)
    ref, _ = _run(m, False, x, t, ctx, mask)
    out, _ = _run(m, True, x, t, ctx, mask)
    assert (out - ref).abs().max().item() < 1e-10


def test_batch_one_is_bit_identical():
    m = _model(torch.float32)
    x, t, ctx, mask = _inputs(torch.float32, 2)
    a, ga = _run(m, False, x[1:], t[1:], ctx[1:], mask[1:])
    b, gb = _run(m, True, x[1:], t[1:], ctx[1:], mask[1:])
    assert torch.equal(a, b)
    assert all(torch.equal(ga[n], gb[n]) for n in ga)
