"""Batch Size > 1 (families/batching.py): the three families that support it (MOSS-Image, Anima, SDXL) at batch 2
and 3 with captions of different lengths, through the dataset's real batching (cache files -> BucketBatchManager ->
the trainer's batch_cond) and each driver's training_loss + backward + an optimizer step; the batch's loss equals the
mean of the same images trained one at a time from the same generator; every other family falls back to 1; DOP and
LR boost switch off above 1; batch 1 is untouched."""
import types

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import save_file  # noqa: E402

from fizgig.dataset.image_dataset import BucketBatchManager, ItemInfo  # noqa: E402
from fizgig.families import batching, launch  # noqa: E402
from fizgig.families.driver import FamilyDriver  # noqa: E402
from fizgig.families.registry import get  # noqa: E402


def _manager(tmp_path, items, batch_size, latent_shape):
    """items: [{cond key: tensor}] -> a BucketBatchManager over real cache files, one bucket, in order."""
    infos = []
    tmp_path.mkdir(parents=True, exist_ok=True)
    for i, cond in enumerate(items):
        lat, te = tmp_path / f"{i}_lat.safetensors", tmp_path / f"{i}_te.safetensors"
        g = torch.Generator().manual_seed(100 + i)
        save_file({f"latent_{latent_shape[-2]}x{latent_shape[-1]}": torch.randn(latent_shape, generator=g)}, str(lat))
        save_file({f"cond__{k}": v.contiguous() for k, v in cond.items()}, str(te))
        it = ItemInfo(f"img{i}", "", (64, 64), (64, 64), latent_cache_path=str(lat))
        it.text_encoder_output_cache_path = str(te)
        infos.append(it)
    return BucketBatchManager({(64, 64): infos}, batch_size)


def _batches(bm):
    return [bm[i] for i in range(len(bm))]


def _check_family(drv, dit, tmp_path, items, latent_shape, batch, tol):
    params = [p for p in dit.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=1e-3)
    one = _batches(_manager(tmp_path / "one", items, 1, latent_shape))
    many = _batches(_manager(tmp_path / "many", items, batch, latent_shape))
    assert [len(b["item_keys"]) for b in many] == ([batch] * (len(items) // batch) +
                                                    ([len(items) % batch] if len(items) % batch else []))
    assert not any(k.endswith("__valid") for b in one for k in b)          # batch 1: no padding, ever
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    with torch.no_grad():
        singles = [drv.training_loss(dit, b["latents"], drv.batch_cond(b, "cpu"), g1)[0].item() for b in one]
    k = 0
    for b in many:
        n = len(b["item_keys"])
        loss, info = drv.training_loss(dit, b["latents"], drv.batch_cond(b, "cpu"), g2)
        want = sum(singles[k:k + n]) / n
        assert abs(loss.item() - want) <= tol * max(1.0, abs(want)), (loss.item(), want)
        loss.backward()                                 # accumulated: the weights the singles saw stay put
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in params)
        k += n
    before = [p.detach().clone() for p in params]
    opt.step()                                          # one optimizer step over the epoch's batches
    assert any(not torch.equal(a, p) for a, p in zip(before, params))
    assert all(torch.isfinite(p).all() for p in params)
    return many


# ---- MOSS-Image ---------------------------------------------------------------------------------------------------
def _moss():
    from fizgig.moss_image.model import MOSSImageDiT
    torch.manual_seed(0)
    dit = MOSSImageDiT(dim=64, ffn_dim=96, num_heads=4, num_layers=3, text_dim=48, freq_dim=32, latent_channels=16)
    for p in dit.parameters():
        p.data.normal_(0, 0.05)
    return dit


@pytest.mark.parametrize("batch", [2, 3])
def test_moss_batches_mixed_captions(tmp_path, batch):
    items = [{"hidden_states": torch.randn(n, 48, generator=torch.Generator().manual_seed(n))} for n in (9, 4, 6, 2, 7)]
    many = _check_family(get("moss_image").load_driver(), _moss(), tmp_path, items, (16, 4, 4), batch, 1e-5)
    assert any("cond__hidden_states__valid" in b for b in many)


# ---- Anima --------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("batch", [2, 3])
def test_anima_batches_mixed_captions(tmp_path, batch):
    pytest.importorskip("einops")
    from fizgig.anima.driver import DIT_CONFIG
    from fizgig.anima.model import Anima
    torch.manual_seed(0)
    dit = Anima(**dict(DIT_CONFIG, model_channels=64, num_blocks=2, num_heads=4, adaln_lora_dim=16)).to(torch.bfloat16)
    for p in dit.parameters():
        p.requires_grad_(p.dim() == 2 and p.shape[0] == 64)       # a few Linears train (stands in for the LoRA)
    g = torch.Generator().manual_seed(3)
    items = [{"qwen": torch.randn(nq, 1024, generator=g), "t5_ids": torch.randint(2, 500, (nt,), generator=g)}
             for nq, nt in ((7, 9), (3, 4), (5, 2), (2, 6), (6, 5))]
    many = _check_family(get("anima").load_driver(), dit, tmp_path, items, (16, 8, 8), batch, 3e-2)  # bf16 model
    assert any("cond__qwen__valid" in b and "cond__t5_ids__valid" in b for b in many)


# ---- SDXL ---------------------------------------------------------------------------------------------------------
class _UNet(torch.nn.Module):
    """Stands in for the diffusers UNet (not in the checks' environment): a per-image function of the latent, the
    timestep, the pooled text and the MEAN over caption rows - like cross-attention, unchanged when every row is
    repeated (pad_conditioning's exact padding), changed by zero rows."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(4, 4) * 0.1)
        self.h = torch.nn.Linear(2048, 4)
        self.p = torch.nn.Linear(1280, 4)

    def forward(self, x, t, encoder_hidden_states, added_cond_kwargs, return_dict=False):
        x = x.float()
        tt = torch.as_tensor(t, dtype=torch.float32).reshape(-1).expand(x.shape[0])
        c = self.h(encoder_hidden_states.float().mean(1)) + self.p(added_cond_kwargs["text_embeds"].float())
        y = torch.einsum("oc,bchw->bohw", self.w, x) + c[:, :, None, None] + (tt / 1000.0)[:, None, None, None]
        return (y,)


@pytest.mark.parametrize("batch", [2, 3])
@pytest.mark.parametrize("min_snr", [0, 5])
def test_sdxl_batches_mixed_captions(tmp_path, batch, min_snr):
    drv = get("sdxl").load_driver()
    drv.set_options({"min_snr": min_snr} if min_snr else {})
    g = torch.Generator().manual_seed(5)
    items = [{"hidden_states": torch.randn(77 * c, 2048, generator=g), "pooled": torch.randn(1280, generator=g)}
             for c in (1, 3, 2, 1, 2)]
    torch.manual_seed(0)
    _check_family(drv, _UNet(), tmp_path, items, (4, 8, 8), batch, 1e-5)


# ---- the rules ----------------------------------------------------------------------------------------------------
def test_which_families_batch():
    assert {k for k in ("moss_image", "anima", "sdxl") if get(k).supports_batch} == {"moss_image", "anima", "sdxl"}
    for k in ("krea2", "qwen_image21", "zimage", "minimax", "acestep", "flux3"):
        if get(k) is not None:
            assert not get(k).supports_batch, k
    assert FamilyDriver.supports_references is False and not hasattr(FamilyDriver, "supports_batch")


class _DS:
    def __init__(self, n, bs, control=False):
        self.batch_size, self.has_control = bs, control
        self.batch_manager = BucketBatchManager({(64, 64): [ItemInfo(f"i{j}", "", (64, 64), (64, 64))
                                                            for j in range(n)]}, bs)

    def __len__(self):
        return len(self.batch_manager)


def _group(*ds):
    g = torch.utils.data.ConcatDataset(list(ds))
    return g


def test_unsupported_family_falls_back_to_one():
    grp = _group(_DS(5, 3))
    assert len(grp) == 2
    assert batching.batch_plan(get("krea2"), grp) == 1
    assert grp.datasets[0].batch_size == 1 and len(grp) == 5               # every image its own step again
    for kw in ({"slider": True}, {"finetune": True}):
        grp = _group(_DS(5, 2))
        assert batching.batch_plan(get("moss_image"), grp, **kw) == 1 and len(grp) == 5
    grp = _group(_DS(5, 2, control=True))
    assert batching.batch_plan(get("sdxl"), grp) == 1                       # edit pairs
    grp = _group(_DS(5, 3))
    assert batching.batch_plan(get("moss_image"), grp) == 3 and len(grp) == 2


def test_batch_one_features_off_above_one():
    flags = dict(log_per_image_loss=True, per_image_lr=True, auto_recaption=True, warmup_look_outliers=True,
                 dop=True, lr_boost=[("x", 2.0)])
    assert batching.batch_features(1, flags) == flags
    off = batching.batch_features(2, flags)
    assert off["lr_boost"] is None and off["dop"] is False                 # DOP and LR boost: batch 1 only
    assert all(off[k] for k in ("log_per_image_loss", "per_image_lr", "auto_recaption", "warmup_look_outliers"))


def _desc(key):
    return get(key)


def test_launch_and_form_batch_size():
    from fizgig.families import form
    moss, krea = _desc("moss_image"), _desc("krea2")
    assert launch.batch_size(moss, {"batch_size": "3"}) == 3
    assert launch.batch_size(krea, {"batch_size": "3"}) == 1
    assert launch.batch_size(moss, {"batch_size": "x"}) == 1
    assert launch.batch_size(moss, {"batch_size": "3", "FAMILY_FT": True}) == (1 if moss.finetune else 3)
    keys = lambda d: {r["key"] for s in form.form(d)["sections"] for r in s["rows"]}
    assert "DATASET_BATCH_SIZE" in keys(moss) and "DATASET_BATCH_SIZE" not in keys(krea)
    st = form.state(moss, {**form.DEFAULTS, "DATASET_BATCH_SIZE": "2"})
    assert not {"KREA2_LOSS_WATCH", "KREA2_PER_IMAGE_LR"} & set(st["disabled"])     # per image at any Batch Size
    assert "DOP" in st["notes"]["DATASET_BATCH_SIZE"] and "per-image" not in st["notes"]["DATASET_BATCH_SIZE"]
    assert "block swap" in form.state(moss, {**form.DEFAULTS, "DATASET_BATCH_SIZE": "4"})["notes"]["DATASET_BATCH_SIZE"]
    st1 = form.state(moss, {**form.DEFAULTS, "DATASET_BATCH_SIZE": "1"})
    assert "DATASET_BATCH_SIZE" not in st1["notes"]
    toml = launch.dataset_toml(moss, {"megapixels": "1.0", "image_folder": "/x", "batch_size": "2"})
    assert "batch_size = 2" in toml
    assert "batch_size = 1" in launch.dataset_toml(krea, {"megapixels": "1.0", "image_folder": "/x",
                                                          "batch_size": "2"})
