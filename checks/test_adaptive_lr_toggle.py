"""Adaptive LR's 'follow validation loss' toggle (Validation tab): form, command, old configs, every optimizer."""
import sys

import torch

from fizgig.training.adaptive_lr import AdaptiveLR
from fizgig.training.optimizers import owns_its_rate


def _setup(tmp_path):
    from fizgig.families import form, launch, registry
    desc = registry.get("moss_image")
    img = tmp_path / "img"
    img.mkdir()
    (img / "a.png").write_bytes(b"x")
    (img / "a.txt").write_text("a photo", encoding="utf-8")
    models = {}
    for mf in desc.model_files:
        if mf.required or mf.pref_key == desc.training_adapter:
            f = tmp_path / f"{mf.pref_key}.safetensors"
            f.write_bytes(b"0")
            models[mf.pref_key] = str(f)

    def cmd(drop=(), **v):
        vals = {**form.DEFAULTS, "LORA_NAME": "t", "LORA_OUTPUT_DIR": str(tmp_path / "out"),
                "DATASET_CONFIG": str(tmp_path / "dataset.toml"), **v}
        for k in drop:
            vals.pop(k, None)
        inp = form.inputs(desc, vals, models=models, python=sys.executable, repo_dir=str(tmp_path),
                          image_folder=str(img), cache_root=str(tmp_path / "cache"),
                          samples_dir=str(tmp_path / "out" / "sample"))
        return launch.plan(desc, inp).stages[-1].cmd
    return form, desc, cmd


def test_form_row_and_greying(tmp_path):
    form, desc, _ = _setup(tmp_path)
    rows = {r["key"]: r for sec in form.form(desc)["sections"] if sec["id"] == "validation" for r in sec["rows"]}
    assert rows["ADAPTIVE_LR_ON_VAL"]["label"] == "Adaptive LR: follow validation loss"
    assert "follows smoothed training loss" in rows["ADAPTIVE_LR_ON_VAL"]["hint"]
    assert rows["ADAPTIVE_LR_ON_VAL"]["kind"] == "check" and form.DEFAULTS["ADAPTIVE_LR_ON_VAL"] is True
    assert "ADAPTIVE_LR_ON_VAL" in form.state(desc, {**form.DEFAULTS, "ADAPTIVE_LR": True})["disabled"]
    assert "ADAPTIVE_LR_ON_VAL" in form.state(desc, {**form.DEFAULTS, "VALIDATION_LOSS": True})["disabled"]
    st = form.state(desc, {**form.DEFAULTS, "VALIDATION_LOSS": True, "ADAPTIVE_LR": True})
    assert "ADAPTIVE_LR_ON_VAL" not in st["disabled"]
    pre = form.to_preset(desc, {**form.DEFAULTS, "ADAPTIVE_LR_ON_VAL": False})      # saved in presets
    assert form.apply_preset(desc, form.DEFAULTS, pre)["ADAPTIVE_LR_ON_VAL"] is False
    assert form.apply_preset(desc, form.DEFAULTS, {})["ADAPTIVE_LR_ON_VAL"] is True  # an old preset: on


def test_command_and_old_configs(tmp_path):
    _, _, cmd = _setup(tmp_path)
    on = dict(VALIDATION_LOSS=True, ADAPTIVE_LR=True)
    assert "--adaptive_lr" in cmd(**on) and "--adaptive_lr_train_loss" not in cmd(**on)
    assert "--adaptive_lr_train_loss" in cmd(**on, ADAPTIVE_LR_ON_VAL=False)
    assert "--adaptive_lr_train_loss" not in cmd(drop=("ADAPTIVE_LR_ON_VAL",), **on)   # an old config: on


def _opt(kind):
    p = [torch.nn.Parameter(torch.ones(4))]
    return {"adamw": lambda: torch.optim.AdamW(p, lr=1e-4), "sgd": lambda: torch.optim.SGD(p, lr=1e-4),
            "adagrad": lambda: torch.optim.Adagrad(p, lr=1e-4)}[kind]()


def test_any_fixed_rate_optimizer_gets_its_rate_written():
    for kind in ("adamw", "sgd", "adagrad"):
        opt = _opt(kind)
        assert not owns_its_rate(opt)
        a = AdaptiveLR(1e-5, 4e-4)
        net = torch.nn.Linear(2, 2)
        a.epoch_boundary(0, 1.0, net, opt, val_loss=1.0)
        lr0 = opt.param_groups[0]["lr"]
        for e, v in enumerate((0.9, 0.8, 0.7), 1):
            a.epoch_boundary(e, 1.0, net, opt, val_loss=v)
        assert opt.param_groups[0]["lr"] != lr0, kind   # the watcher drives a plain torch optimizer


def test_self_rate_optimizers_stand_down():
    p = [torch.nn.Parameter(torch.ones(2))]
    o = torch.optim.SGD(p, lr=1.0)
    o.param_groups[0]["d"] = 1e-6          # DAdapt / Prodigy style
    assert owns_its_rate(o)
