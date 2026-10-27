"""The loss graph's four trend views: epoch averages, the kohya-style running average and the fixed validation loss (deterministic, no grad, training state untouched)."""
import json, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fizgig.training.step_log import StepLog, read, loss_path, epoch_means, running_avg  # noqa: E402
from fizgig.training.val_loss import FixedValidation, fixed_timesteps, pick_keys  # noqa: E402


def _rows(losses_by_epoch):
    rows, s = [], 0
    for e, ls in enumerate(losses_by_epoch, 1):
        for l in ls:
            s += 1
            rows.append([s, e, l, None, None])
    return rows


def test_epoch_means():
    assert epoch_means(_rows([[1, 3], [2, 2, 2]])) == [[2, 1, 2.0], [5, 2, 2.0]]


def test_running_avg_kohya():
    r = running_avg(_rows([[4, 2], [1, 1]]))
    # epoch 1: cumulative 4, 3; epoch 2 replaces position 0 then 1: (1+2)/2, (1+1)/2
    assert [x[1] for x in r] == [4.0, 3.0, 1.5, 1.0]


def test_val_lines_and_old_logs(tmp_path):
    lg = StepLog(str(tmp_path), "x")
    for i in range(1, 5):
        lg.add(i, 1, 0.1 * i, lr=1e-4)
    lg.add_validation(4, 1, 0.05, [0.04, 0.06], 1.2)
    lg.flush()
    d = read(loss_path(str(tmp_path), "x"))
    assert len(d["points"]) == 4 and len(d["points"][0]) == 4 and "bands" not in d
    assert d["val"] == [[4, 1, 0.05, [0.04, 0.06]]]
    assert "t" not in json.loads(open(loss_path(str(tmp_path), "x")).readline())   # step lines as before
    p = loss_path(str(tmp_path), "old")
    open(p, "w").write('{"s": 1, "e": 1, "l": 0.08, "lr": 0.0001}\n{"s": 2, "e": 1, "l": 0.07}\n')
    d = read(p)
    assert d["val"] == [] and len(d["epoch_mean"]) == 1


def test_resume_keeps_val_lines(tmp_path):
    lg = StepLog(str(tmp_path), "r")
    lg.add(1, 1, 0.1); lg.add_validation(1, 1, 0.2); lg.add(2, 1, 0.1); lg.add_validation(2, 1, 0.2); lg.flush()
    d = read(StepLog(str(tmp_path), "r", start_step=1).path)
    assert len(d["points"]) == 1 and len(d["val"]) == 1


def test_pick_and_timesteps():
    assert fixed_timesteps(4) == [0.125, 0.375, 0.625, 0.875]
    keys = [f"img{i:02d}" for i in range(10)]
    assert pick_keys(reversed(keys), 4) == pick_keys(keys, 4) == ["img00", "img03", "img06", "img09"]
    assert pick_keys(keys, 50) == sorted(keys)


class _Dit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)
        self.drop = torch.nn.Dropout(0.5)

    def forward(self, x):
        return self.drop(self.lin(x))


class _Driver:
    def batch_cond(self, batch, device):
        return {"c": batch["cond__c"].to(device)}

    def training_loss(self, dit, latents, cond, generator, *, min_t=0.0, max_t=1.0, refs=None):
        t = min_t + (max_t - min_t) * torch.rand(1, generator=generator)
        noise = torch.randn(latents.shape, generator=generator)
        xt = (1 - t) * latents + t * noise
        assert not torch.is_grad_enabled()
        return torch.nn.functional.mse_loss(dit(xt + cond["c"]), noise - latents), {"t": float(t)}


def test_validation_deterministic_and_harmless():
    torch.manual_seed(0)
    dit, drv = _Dit(), _Driver()
    dit.train()
    v = FixedValidation([f"k{i}" for i in range(6)], images=3, timesteps=4, seed=7)
    for i in range(6):
        v.offer({"item_keys": [f"k{i}"], "latents": torch.randn(1, 4), "cond__c": torch.randn(1, 4)})
    assert v.ready and len(v.batches) == 3
    rng = torch.get_rng_state()
    w = dit.lin.weight.clone()
    a = v.run(drv, dit, "cpu")
    b = v.run(drv, dit, "cpu")
    assert a[0] == b[0] and a[1] == b[1] and len(a[1]) == 4
    assert torch.equal(torch.get_rng_state(), rng)           # training's random stream is untouched
    assert dit.training and dit.lin.weight.grad is None and torch.equal(dit.lin.weight, w)
    v2 = FixedValidation([f"k{i}" for i in range(6)], images=3, timesteps=4, seed=8)
    v2.batches = v.batches
    assert v2.run(drv, dit, "cpu")[0] != a[0]                # the seed is the noise


# ---- picked validation images ---------------------------------------------------------------------------------
from fizgig.training.val_pick import picked_list, resolve_picked  # noqa: E402


def test_picked_paths_resolve_to_cache_keys():
    keys = ["img01", "img02", "photo.v2", "C:/data/set/face 3.png", "clip7#still"]
    found, missing = resolve_picked(["img02.png", "photo.v2.jpg", "face 3.png", "IMG01.PNG", "gone.png", "clip7.mp4"],
                                    keys)
    assert found == ["img02", "photo.v2", "C:/data/set/face 3.png", "img01", "clip7#still"]
    assert missing == ["gone.png"]


def test_picked_list_forms():
    assert picked_list(None) == picked_list([]) == picked_list("") == []
    assert picked_list("a.png\nb.png; a.png") == ["a.png", "b.png"]
    assert picked_list([" a.png ", "", "a.png"]) == ["a.png"]


def test_picked_used_and_fallback(caplog):
    keys = [f"k{i}" for i in range(6)]
    v = FixedValidation(keys, images=2, picked=["k5.png", "nope.png"])
    assert v.wanted == ["k5"] and v.missing == ["nope.png"]
    assert any("nope.png" in r.message and "skipped" in r.message for r in caplog.records)
    assert FixedValidation(keys, images=2, picked=[]).wanted == pick_keys(keys, 2)          # none picked: auto
    assert FixedValidation(keys, images=2, picked=["x.png"]).wanted == pick_keys(keys, 2)   # none found: auto


def test_picked_setting_round_trip(tmp_path):
    """Saved settings (JSON) -> the form greys the count -> the launch plan passes each picked name to the trainer;
    none picked: the count goes instead and no --validation_pick."""
    from fizgig.families import form, launch, registry
    krea = registry.get("krea2")
    assert form.DEFAULTS["VALIDATION_PICKED"] == []
    names = ["b.png", "a b.jpg"]
    saved = json.loads(json.dumps({**form.DEFAULTS, "VALIDATION_LOSS": True, "VALIDATION_PICKED": names}))
    assert saved["VALIDATION_PICKED"] == names
    st = form.state(krea, saved)
    assert "VALIDATION_IMAGES" in st["disabled"]
    assert "VALIDATION_IMAGES" not in form.state(krea, {**saved, "VALIDATION_PICKED": []})["disabled"]
    img = tmp_path / "imgs"; img.mkdir()
    (img / "b.png").write_bytes(b"x"); (img / "b.txt").write_text("zwxem a photo", encoding="utf-8")
    models = {}
    for mf in krea.model_files:
        if mf.required or mf.pref_key == krea.training_adapter:
            f = tmp_path / f"{mf.pref_key}.safetensors"; f.write_bytes(b"0"); models[mf.pref_key] = str(f)

    def cmd(vals):
        inp = form.inputs(krea, {**vals, "LORA_NAME": "v", "LORA_OUTPUT_DIR": str(tmp_path / "out"),
                                 "DATASET_CONFIG": str(tmp_path / "dataset.toml")},
                          models=models, python=sys.executable, repo_dir=str(tmp_path), image_folder=str(img),
                          cache_root=str(tmp_path / "cache"), samples_dir=str(tmp_path / "out" / "sample"))
        pl = launch.plan(krea, inp)
        assert pl.problems == []
        return pl.stages[-1].cmd

    c = cmd(saved)
    assert "--validation_loss" in c
    assert [c[i + 1] for i, t in enumerate(c) if t == "--validation_pick"] == names
    c = cmd({**saved, "VALIDATION_PICKED": []})
    assert "--validation_pick" not in c and c[c.index("--validation_images") + 1] == "4"
    assert "--validation_loss" not in cmd({**saved, "VALIDATION_LOSS": False})


def test_validation_rows_live_in_samples_and_stay_in_presets():
    from fizgig.families import form, registry
    krea = registry.get("krea2")
    secs = form.form(krea)["sections"]
    sec = next(s for s in secs if s["id"] == "validation")          # its own collapsible card on the Samples tab
    assert sec["title"] == "Image Validation" and sec["tab"] == "samples" and sec["collapsed"] is True
    samples = [s["id"] for s in secs if s["tab"] == "samples"]
    assert samples.index("validation") > samples.index("samples")  # after the preview cards
    assert [r["key"] for r in sec["rows"]] == ["VALIDATION_LOSS", "VALIDATION_PICKED", "VALIDATION_IMAGES",
                                               "VALIDATION_TIMESTEPS", "VALIDATION_SEED", "VALIDATION_EVERY_STEPS",
                                               "VALIDATION_EARLY_STOP", "VALIDATION_PATIENCE"]
    assert sec["rows"][0]["full"] is True                          # the switch spans the card: wide and short
    keys = [r["key"] for s in secs if s["id"] != "validation" for r in s["rows"]]
    assert not any(k.startswith("VALIDATION_") or k == "_validation_heading" for k in keys)
    vals = {**form.DEFAULTS, "VALIDATION_LOSS": True, "VALIDATION_PICKED": ["a.png"], "VALIDATION_SEED": "7"}
    pre = form.to_preset(krea, vals)                               # presets carry them as before
    assert pre["VALIDATION_LOSS"] is True and pre["VALIDATION_PICKED"] == ["a.png"] and pre["VALIDATION_SEED"] == "7"
    back = form.apply_preset(krea, form.DEFAULTS, pre)
    assert back["VALIDATION_PICKED"] == ["a.png"] and back["VALIDATION_SEED"] == "7"


def test_how_often_to_check_mapping():
    from fizgig.training.val_pick import validation_every_steps
    from fizgig.families import form, registry
    assert [validation_every_steps(v) for v in ("0", 0, "", None, "500", " 200 ", "-5", "abc", "1.5")] == \
        [0, 0, 0, 0, 500, 200, 0, 0, 0]
    krea = registry.get("krea2")
    row = next(r for s in form.form(krea)["sections"] for r in s["rows"] if r and r["key"] == "VALIDATION_EVERY_STEPS")
    assert row["kind"] == "valevery" and row["default"] == "0" and row["n_default"] == "500"
    st = lambda v: form.state(krea, {**form.DEFAULTS, "VALIDATION_LOSS": True, "VALIDATION_EVERY_STEPS": v})
    assert "VALIDATION_EVERY_N" in st("0")["disabled"]                     # an old run's 0: Every epoch
    assert "VALIDATION_EVERY_N" not in st("300")["disabled"]               # an old run's N: Every N steps
    assert "VALIDATION_EVERY_STEPS" in st("-2")["notes"]


def test_how_often_to_check_launch(tmp_path):
    from fizgig.families import form, launch, registry
    krea = registry.get("krea2")
    img = tmp_path / "imgs"; img.mkdir()
    (img / "b.png").write_bytes(b"x"); (img / "b.txt").write_text("zwxem a photo", encoding="utf-8")
    models = {}
    for mf in krea.model_files:
        if mf.required or mf.pref_key == krea.training_adapter:
            f = tmp_path / f"{mf.pref_key}.safetensors"; f.write_bytes(b"0"); models[mf.pref_key] = str(f)

    def every(v):
        vals = {**form.DEFAULTS, "VALIDATION_LOSS": True, "LORA_NAME": "v", "LORA_OUTPUT_DIR": str(tmp_path / "out"),
                "DATASET_CONFIG": str(tmp_path / "dataset.toml")}
        if v is not None:
            vals["VALIDATION_EVERY_STEPS"] = v
        inp = form.inputs(krea, vals, models=models, python=sys.executable, repo_dir=str(tmp_path),
                          image_folder=str(img), cache_root=str(tmp_path / "cache"),
                          samples_dir=str(tmp_path / "out" / "sample"))
        c = launch.plan(krea, inp).stages[-1].cmd
        return c[c.index("--validation_every_steps") + 1]

    assert every(None) == "0" and every("0") == "0" and every("500") == "500" and every("-5") == "0"
