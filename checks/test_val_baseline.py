"""The step-0 validation baseline (Chris, 29 Oct 2026): the untrained LoRA's validation loss is the line's first point
and Adaptive LR's first best, so epoch 1 is judged against it - with no LR change at step 0."""
import logging

import torch

from fizgig.families import train as T
from fizgig.training.adaptive_lr import AdaptiveLR


class _Validator:
    def __init__(self, ready=True):
        self.ts, self.seg, self.batches, self._ready = [0.25, 0.75], 0, {"a": 1}, ready

    def prefill(self, datasets):
        return self._ready


class _Log:
    def __init__(self):
        self.rows = []

    def add_validation(self, *a, **k):
        self.rows.append(a)


def test_baseline_logs_plots_and_seeds_adaptive_lr(caplog):
    a, log = AdaptiveLR(1e-5, 2e-4), _Log()
    with caplog.at_level(logging.INFO):
        ran = T._baseline(_Validator(), a, [], log, 0, 0, lambda: ((0.1360, [0.1, 0.17], 2.0), False))
    assert ran and a.best_val == 0.1360 and log.rows[0][:3] == (0, 0, 0.1360)
    assert "step 0 (baseline): loss 0.13600" in caplog.text


def test_no_baseline_when_images_are_not_ready_or_the_check_fails():
    a, log = AdaptiveLR(1e-5, 2e-4), _Log()
    assert not T._baseline(_Validator(ready=False), a, [], log, 0, 0, lambda: 1 / 0)
    assert not T._baseline(_Validator(), a, [], log, 0, 0, lambda: (None, True))
    assert a.best_val is None and not log.rows


def test_epoch_one_is_judged_against_the_baseline_without_an_lr_change():
    net = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(net.parameters(), lr=8.9e-5)
    a = AdaptiveLR(1e-5, 2e-4)
    a.set_baseline(0.1380)
    a.epoch_boundary(0, 0.143, net, opt, val_loss=0.1360)       # better than the untrained LoRA
    assert opt.param_groups[0]["lr"] == 8.9e-5 and a.good_streak == 1 and a.best_val == 0.1360
    a.epoch_boundary(1, 0.142, net, opt, val_loss=0.1350)       # a second better epoch: probes up already
    assert opt.param_groups[0]["lr"] > 8.9e-5


def test_a_resumed_best_is_kept():
    a = AdaptiveLR(1e-5, 2e-4)
    a.best_val = 0.1335
    a.set_baseline(0.1400)
    assert a.best_val == 0.1335


def test_prefill_reads_the_picked_images_from_the_cache(tmp_path):
    from types import SimpleNamespace

    from safetensors.torch import save_file

    from fizgig.dataset.image_dataset import ItemInfo
    from fizgig.training.val_loss import FixedValidation
    items = []
    for k in ("a", "b", "c"):
        it = ItemInfo(k, "cap", (64, 64), (64, 64), latent_cache_path=str(tmp_path / f"{k}_l.safetensors"))
        it.text_encoder_output_cache_path = str(tmp_path / f"{k}_t.safetensors")
        save_file({"latent_8x8_bf16": torch.zeros(4, 8, 8)}, it.latent_cache_path)
        save_file({"cond__x": torch.ones(3)}, it.text_encoder_output_cache_path)
        items.append(it)
    ds = SimpleNamespace(batch_manager=SimpleNamespace(buckets={(64, 64): items}))
    v = FixedValidation(["a", "b", "c"], images=2, picked=["c"])
    assert v.prefill([ds]) and list(v.batches) == ["c"]
    assert v.batches["c"]["item_keys"] == ["c"] and v.batches["c"]["latents"].shape[0] == 1
