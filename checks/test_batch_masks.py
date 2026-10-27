"""Masked training at Batch Size > 1 (MOSS-Image, Anima, SDXL): a batch's loss with masks (and masked prior
preservation) equals the mean of the same images trained one at a time from the same generator - each image's mask
scaled to mean 1 on its own; an image without a mask (all ones in the batch) trains as normal and gets no prior."""
import pytest

torch = pytest.importorskip("torch")

from checks.test_batch_size import _UNet, _batches, _manager, _moss  # noqa: E402
from fizgig.families.registry import get  # noqa: E402


class _Prior:
    weight = 0.5

    def base(self, forward):          # stands in for the LoRA at 0: the forward, a little off
        with torch.no_grad():
            return forward().detach().float() * 0.9


def _masks(n, hw, seed=11):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        out.append(torch.ones(1, *hw) if i % 3 == 1 else (torch.rand(1, *hw, generator=g) > 0.5).float() * 0.9 + 0.1)
    return out


def _check(drv, dit, tmp_path, items, latent_shape, batch, tol):
    hw = latent_shape[-2:]
    masks = _masks(len(items), hw)
    one = _batches(_manager(tmp_path / "one", items, 1, latent_shape))
    many = _batches(_manager(tmp_path / "many", items, batch, latent_shape))
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    singles = []
    with torch.no_grad():
        for i, b in enumerate(one):
            kw = {} if i % 3 == 1 else {"loss_mask": masks[i][None], "prior": _Prior()}
            singles.append(drv.training_loss(dit, b["latents"], drv.batch_cond(b, "cpu"), g1, **kw)[0].item())
    k = 0
    for b in many:
        n = len(b["item_keys"])
        lm = torch.stack(masks[k:k + n])
        loss, _ = drv.training_loss(dit, b["latents"], drv.batch_cond(b, "cpu"), g2, loss_mask=lm, prior=_Prior())
        want = sum(singles[k:k + n]) / n
        assert abs(loss.item() - want) <= tol * max(1.0, abs(want)), (loss.item(), want)
        loss.backward()
        k += n


@pytest.mark.parametrize("batch", [2, 3])
def test_moss_batches_masks(tmp_path, batch):
    items = [{"hidden_states": torch.randn(n, 48, generator=torch.Generator().manual_seed(n))} for n in (9, 4, 6, 2, 7)]
    _check(get("moss_image").load_driver(), _moss(), tmp_path, items, (16, 4, 4), batch, 1e-5)


@pytest.mark.parametrize("batch", [2, 3])
@pytest.mark.parametrize("min_snr", [0, 5])
def test_sdxl_batches_masks(tmp_path, batch, min_snr):
    drv = get("sdxl").load_driver()
    drv.set_options({"min_snr": min_snr} if min_snr else {})
    g = torch.Generator().manual_seed(5)
    items = [{"hidden_states": torch.randn(77 * c, 2048, generator=g), "pooled": torch.randn(1280, generator=g)}
             for c in (1, 3, 2, 1, 2)]
    torch.manual_seed(0)
    _check(drv, _UNet(), tmp_path, items, (4, 8, 8), batch, 1e-5)


@pytest.mark.parametrize("batch", [2, 3])
def test_anima_batches_masks(tmp_path, batch):
    pytest.importorskip("einops")
    from fizgig.anima.driver import DIT_CONFIG
    from fizgig.anima.model import Anima
    torch.manual_seed(0)
    dit = Anima(**dict(DIT_CONFIG, model_channels=64, num_blocks=2, num_heads=4, adaln_lora_dim=16)).to(torch.bfloat16)
    for p in dit.parameters():
        p.requires_grad_(p.dim() == 2 and p.shape[0] == 64)
    g = torch.Generator().manual_seed(3)
    items = [{"qwen": torch.randn(nq, 1024, generator=g), "t5_ids": torch.randint(2, 500, (nt,), generator=g)}
             for nq, nt in ((7, 9), (3, 4), (5, 2), (2, 6), (6, 5))]
    _check(get("anima").load_driver(), dit, tmp_path, items, (16, 8, 8), batch, 3e-2)
