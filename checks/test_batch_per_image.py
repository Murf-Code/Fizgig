"""Per-image features at Batch Size > 1 (MOSS-Image, Anima, SDXL): the batched driver hands back each image's own
loss (info["per"]), equal to what that image's batch-1 step gives from the same generator (masks and masked prior
included); weigh_batch gives each image its own weight - per-image LR, the look warm-up, a regularisation image's
nudge - and leaves an excluded image out, so the weighted batch loss (and its gradient) is the mean of the weighted
batch-1 losses; the loss watch records every image of the batch on its own, and its verdicts and multipliers are per
image; batch 1 is unchanged; the launch sends the switches at any Batch Size, Prodigy too."""
import os
import sys

import pytest

torch = pytest.importorskip("torch")

from checks.test_batch_masks import _masks, _Prior  # noqa: E402
from checks.test_batch_size import _UNet, _batches, _manager, _moss  # noqa: E402
from fizgig.families import batching  # noqa: E402
from fizgig.families.loss_watch import Watch, weigh_batch  # noqa: E402
from fizgig.families.registry import get  # noqa: E402
from fizgig.training.loss_logger import PerImageLossWatch  # noqa: E402

MOSS_ITEMS = lambda: [{"hidden_states": torch.randn(n, 48, generator=torch.Generator().manual_seed(n))}
                      for n in (9, 4, 6, 2, 7)]


class _W:
    """A Watch around a real PerImageLossWatch, without a dataset (Watch.__init__ needs a group)."""
    def __init__(self, tmp_path, per_image_lr=True):
        self.watch = PerImageLossWatch(str(tmp_path / "out"), apply_lr=per_image_lr, write_jsonl=True)
    excluded, multipliers, excluded_keys, observe, observe_each, multiplier = (
        Watch.excluded, Watch.multipliers, Watch.excluded_keys, Watch.observe, Watch.observe_each, Watch.multiplier)


def _per_image(drv, dit, tmp_path, items, shape, batch, tol, masked):
    """Each image's batched loss vs its batch-1 loss; returns (singles, [(batch, loss, info)])."""
    masks = _masks(len(items), shape[-2:])
    one = _batches(_manager(tmp_path / "one", items, 1, shape))
    many = _batches(_manager(tmp_path / "many", items, batch, shape))
    kw1 = lambda i: ({"loss_mask": masks[i][None], "prior": _Prior()} if masked and i % 3 != 1 else {})
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    with torch.no_grad():
        singles = []
        for i, b in enumerate(one):
            l1, info1 = drv.training_loss(dit, b["latents"], drv.batch_cond(b, "cpu"), g1, **kw1(i))
            assert "per" not in info1                                          # batch 1: as before
            singles.append(l1.item())
    out, k = [], 0
    for b in many:
        n = len(b["item_keys"])
        kw = {"loss_mask": torch.stack(masks[k:k + n]), "prior": _Prior()} if masked else {}
        loss, info = drv.training_loss(dit, b["latents"], drv.batch_cond(b, "cpu"), g2, **kw)
        if n == 1:                                       # a bucket's last, smaller batch of one: the batch-1 path
            assert "per" not in info and abs(loss.item() - singles[k]) <= tol * max(1.0, abs(singles[k]))
            k += 1
            continue
        per = info["per"]
        assert per.shape == (n,) and len(info["ts"]) == n
        assert abs(per.mean().item() - loss.item()) <= 1e-6 * max(1.0, abs(loss.item()))
        for j in range(n):
            assert abs(per[j].item() - singles[k + j]) <= tol * max(1.0, abs(singles[k + j])), (j, per[j], singles)
        out.append((b, loss, info))
        k += n
    return singles, out


@pytest.mark.parametrize("masked", [False, True])
def test_moss_per_image_losses(tmp_path, masked):
    _per_image(get("moss_image").load_driver(), _moss(), tmp_path, MOSS_ITEMS(), (16, 4, 4), 2, 1e-5, masked)


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("min_snr", [0, 5])
def test_sdxl_per_image_losses(tmp_path, masked, min_snr):
    drv = get("sdxl").load_driver()
    drv.set_options({"min_snr": min_snr} if min_snr else {})
    g = torch.Generator().manual_seed(5)
    items = [{"hidden_states": torch.randn(77 * c, 2048, generator=g), "pooled": torch.randn(1280, generator=g)}
             for c in (1, 3, 2, 1)]
    torch.manual_seed(0)
    _per_image(drv, _UNet(), tmp_path, items, (4, 8, 8), 2, 1e-5, masked)


@pytest.mark.parametrize("masked", [False, True])
def test_anima_per_image_losses(tmp_path, masked):
    pytest.importorskip("einops")
    from fizgig.anima.driver import DIT_CONFIG
    from fizgig.anima.model import Anima
    torch.manual_seed(0)
    dit = Anima(**dict(DIT_CONFIG, model_channels=64, num_blocks=2, num_heads=4, adaln_lora_dim=16)).to(torch.bfloat16)
    g = torch.Generator().manual_seed(3)
    items = [{"qwen": torch.randn(nq, 1024, generator=g), "t5_ids": torch.randint(2, 500, (nt,), generator=g)}
             for nq, nt in ((7, 9), (3, 4), (5, 2), (2, 6))]
    _per_image(get("anima").load_driver(), dit, tmp_path, items, (16, 8, 8), 2, 3e-2, masked)


@pytest.mark.parametrize("masked", [False, True])
def test_weighted_batch_equals_mean_of_weighted_singles(tmp_path, masked):
    """Batch 2, per-image LR on: loss and gradient match the batch-1 steps each weighted by its own multiplier."""
    drv, dit = get("moss_image").load_driver(), _moss()
    items = MOSS_ITEMS()[:4]
    w = _W(tmp_path)
    w.watch._mult = {"img0": 0.5, "img1": 1.1, "img3": 0.1}                    # img2 stays at 1.0
    w.watch.set_warmup_keys({"img1"}, start=0.4, ramp_epochs=4)                # epoch 1: x0.4 (below its 1.1)
    want_m = {"img0": 0.5, "img1": 0.4, "img2": 1.0, "img3": 0.1}
    masks = _masks(4, (4, 4))
    params = [p for p in dit.parameters() if p.requires_grad]
    one = _batches(_manager(tmp_path / "one", items, 1, (16, 4, 4)))
    g1 = torch.Generator().manual_seed(7)
    singles = []
    for i, b in enumerate(one):
        kw = {"loss_mask": masks[i][None]} if masked else {}
        l1, _ = drv.training_loss(dit, b["latents"], drv.batch_cond(b, "cpu"), g1, **kw)
        m = w.multiplier(b)
        assert m == pytest.approx(want_m[b["item_keys"][0]])
        (l1 * m / 2).backward()                                                # two batches of 2 -> mean over 2
        singles.append((l1 * m).item())
    ref = [p.grad.detach().clone() for p in params]
    for p in params:
        p.grad = None
    g2 = torch.Generator().manual_seed(7)
    k = 0
    for b in _batches(_manager(tmp_path / "many", items, 2, (16, 4, 4))):
        kw = {"loss_mask": torch.stack(masks[k:k + 2])} if masked else {}
        loss, info = drv.training_loss(dit, b["latents"], drv.batch_cond(b, "cpu"), g2, **kw)
        scaled, skip = weigh_batch(loss, info, b["item_keys"], w)
        assert not skip
        assert scaled.item() == pytest.approx(sum(singles[k:k + 2]) / 2, rel=1e-5)
        scaled.backward()
        k += 2
    for a, r in zip([p.grad for p in params], ref):
        assert torch.allclose(a, r, rtol=1e-4, atol=1e-7)


def test_exclusion_and_reg_in_a_batch(tmp_path):
    w = _W(tmp_path)
    w.watch._excluded = {"a"}
    w.watch._mult = {"b": 0.5}
    per = torch.tensor([2.0, 4.0, 6.0], requires_grad=True)
    loss = per.mean()
    scaled, skip = weigh_batch(loss, {"per": per}, ["a", "b", "r"], w, reg_keys={"r"}, reg_mult=0.25)
    assert skip == {0}
    assert scaled.item() == pytest.approx((4.0 * 0.5 + 6.0 * 0.25) / 2)        # 'a' sits out: mean over the 2 kept
    assert not w.excluded({"item_keys": ["a", "b"]}) and w.excluded({"item_keys": ["a"]})
    w.watch._excluded = {"a", "b"}
    assert w.excluded({"item_keys": ["a", "b"]})                               # every image out: the step is skipped
    w.observe_each(1, 1, ["a", "b", "c"], [0.1, 0.5, 0.9], [1.0, 2.0, 3.0], skip={0, 1})
    assert [r[0] for r in w.watch._records] == ["c"]
    # all at x1: the batch's own loss, untouched; no per (batch 1 / a slider): None -> the batch-1 path
    w2 = _W(tmp_path / "2")
    assert weigh_batch(loss, {"per": per}, ["x", "y", "z"], w2)[0] is loss
    assert weigh_batch(loss, {"t": 0.5}, ["x"], w2) is None
    assert weigh_batch(loss, {"per": per[:1]}, ["x"], w2) is None


def test_watch_records_every_image_and_gives_per_image_verdicts(tmp_path):
    """Batches of 2 recorded image by image: no composite keys, per-image LR stays on, verdicts are per image and
    their multipliers reach weigh_batch. 'bad' stays far above the rest every epoch."""
    import random
    rnd = random.Random(0)
    w = _W(tmp_path)
    keys = [f"k{i}" for i in range(9)] + ["bad"]
    step = 0
    for ep in range(1, 13):
        order = keys[:]
        rnd.shuffle(order)
        for j in range(0, len(order), 2):
            pair = order[j:j + 2]
            ts = [rnd.random() for _ in pair]
            losses = [(0.9 if k == "bad" else 0.1 + 0.02 * rnd.random()) for k in pair]
            step += 1
            w.observe_each(ep, step, pair, ts, losses)
        w.watch.epoch_boundary(ep)
    assert not w.watch._batched
    rec = {r[0] for r in w.watch._records}
    assert rec == set(keys) and not any("|" in k for k in rec)
    assert len(w.watch._records) == 12 * len(keys)                              # one per image per epoch
    assert set(w.watch.verdicts) <= set(keys) and w.watch.verdicts.get("bad") not in (None, "ok", "healthy")
    m = w.multipliers(["bad", "k0"])
    assert m[0] < 1.0
    per = torch.tensor([1.0, 1.0])
    scaled, _ = weigh_batch(per.mean(), {"per": per}, ["bad", "k0"], w)
    assert scaled.item() == pytest.approx((m[0] + m[1]) / 2)
    # the JSONL holds one line per image, so Problem Images and a resume read them as at batch 1
    import json
    lines = [json.loads(x) for x in open(os.path.join(tmp_path, "out", "loss_log", "per_image_loss.jsonl"),
                                         encoding="utf-8")]
    assert len(lines) == 12 * len(keys) and not any("|" in str(x.get("key", x.get("item_keys", ""))) for x in lines)


def test_batch_features_keep_the_loss_watch():
    flags = dict(log_per_image_loss=True, per_image_lr=True, auto_recaption=True, warmup_look_outliers=True,
                 dop=True, lr_boost=[("x", 2.0)])
    two = batching.batch_features(2, flags)
    assert two["per_image_lr"] and two["log_per_image_loss"] and two["auto_recaption"] and two["warmup_look_outliers"]
    assert two["dop"] is False and two["lr_boost"] is None                     # still batch-1 only
    assert batching.batch_features(1, flags) == flags


def test_trainer_wires_it():
    src = open(os.path.join(os.path.dirname(batching.__file__), "train.py"), encoding="utf-8").read()
    assert "weigh_batch(loss, _info, batch.get(\"item_keys\"), watch" in src
    assert "watch.observe_each(" in src
