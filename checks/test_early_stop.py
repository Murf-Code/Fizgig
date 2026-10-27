"""Early stopping on the fixed validation loss: the counter itself, a tiny CPU run whose validation plateaus and
stops early with its best check's LoRA as the final one (each check and checks every N steps, through Save & stop
and Resume), a run that keeps improving and runs to the end, and the form, the command line, the loss graph's
marker and the server's run-card note."""
import json
import os
import sys

from safetensors import safe_open
from safetensors.torch import load_file

from fizgig.training.early_stop import EarlyStop
from test_save_stop import ROOT, SPE, _losses, _run, data, tiny  # noqa: F401  (the tiny CPU family, its fixtures)

PLATEAU = dict(validation_loss=True, validation_images=2, validation_early_stop=True, validation_patience=2,
               max_train_epochs=10, learning_rate=1e-9, lr_scheduler="constant")


def _same(a, b):
    x, y = load_file(str(a)), load_file(str(b))
    return x.keys() == y.keys() and all((x[k] == y[k]).all() for k in x)


def _epoch_meta(path):
    with safe_open(str(path), "pt") as f:
        return f.metadata().get("ss_epoch")


def test_counter():
    es = EarlyStop(patience=2, min_delta=0.001)
    assert es.observe(1.0, 10, 1) == "best"
    assert es.observe(0.9995, 20, 2) == "wait"          # under 0.1% better: noise, not progress
    assert es.observe(0.95, 30, 3) == "best" and es.bad == 0 and es.best_step == 30
    assert es.observe(0.96, 40, 4) == "wait"
    assert es.observe(0.951, 50, 5) == "stop" and es.stopped
    again = EarlyStop(patience=2)
    assert again.load_state_dict(es.state_dict()) is False  # a Continue after it stopped: counts afresh
    assert again.bad == 0 and again.best == 0.95 and not again.stopped
    mid = EarlyStop(patience=3)
    mid.observe(1.0, 1, 1)
    mid.observe(1.0, 2, 1)
    back = EarlyStop(patience=3)
    assert back.load_state_dict(mid.state_dict()) and back.bad == 1      # a resume carries the count on
    assert back.observe(5.0, 3, 1, seg=1) == "best"     # validation changed what it checks: a new best


def test_plateau_stops_early_with_the_best_lora_as_final(tiny, tmp_path, caplog):
    T, cfg = tiny
    out = tmp_path / "plateau"
    caplog.set_level("INFO")
    assert _run(T, cfg, out, **PLATEAU) == "done"
    # checks at the end of epochs 1, 2, 3: the best is epoch 1's, then 2 without better -> stop after epoch 3
    assert "[early stop] no better for 2 checks: stopping at step 12. Best was step 4 (epoch 1)" in caplog.text
    assert "[early stop] Final LoRA = best validation, step 4 (epoch 1): t.safetensors" in caplog.text
    assert _same(out / "t.safetensors", out / "t-000001.safetensors") and _epoch_meta(out / "t.safetensors") == "1"
    assert not _same(out / "t.safetensors", out / "t-000003.safetensors")
    assert (out / "t-000003.safetensors").exists() and not (out / "t-000004.safetensors").exists()  # normal saves
    assert not (out / "t-best.safetensors").exists()                # the final is the best: not kept twice
    assert not (out / "t-000010.safetensors").exists()
    assert len(_losses(out)) == 3 * SPE
    from fizgig.training import step_log
    g = step_log.read(step_log.loss_path(str(out), "t"))
    assert g["stop"] == {"step": 12, "epoch": 3, "best_step": 4, "best_epoch": 1, "best": g["stop"]["best"]}


def test_every_n_steps_through_save_and_stop_and_resume(tiny, tmp_path, caplog):
    T, cfg = tiny
    out = tmp_path / "steps"
    args = {**PLATEAU, "validation_every_steps": 2, "validation_patience": 3}
    assert _run(T, cfg, out, press_at=SPE + 2, **args) == "stopped"   # Save & stop part-way through the count
    es = json.loads((out / "t-000002-state" / "training_state.json").read_text())["early_stop"]
    best, bad = es["best_step"], es["bad"]
    assert best <= SPE and bad == (SPE + 2 - best) // 2 and 0 < bad < 3   # the restore point holds the count
    assert (out / "t-best.safetensors").exists()                     # its step has no numbered LoRA: kept on its own
    os.remove(out / ".pause_requested")
    caplog.set_level("INFO")
    assert _run(T, cfg, out, resume=str(out / "t-000002-state"), **args) == "done"
    assert f"no better for {bad + 1}/3 checks" in caplog.text or bad + 1 == 3   # carried on from the restore point
    stop = best + 3 * 2
    assert f"stopping at step {stop}. Best was step {best} (epoch 1)" in caplog.text
    assert [r["s"] for r in _losses(out)] == list(range(1, stop + 1))
    # stopped inside an epoch: no numbered save for it; at an epoch's last step: that epoch's save as usual
    assert (out / f"t-{-(-stop // SPE):06d}.safetensors").exists() == (stop % SPE == 0)
    assert _epoch_meta(out / "t.safetensors") == "1" and not (out / "t-best.safetensors").exists()


def test_still_improving_runs_to_the_end_and_keeps_the_best(tiny, tmp_path, caplog):
    T, cfg = tiny
    out = tmp_path / "better"
    caplog.set_level("INFO")
    assert _run(T, cfg, out, validation_loss=True, validation_images=2, validation_early_stop=True,
                validation_patience=2, learning_rate=1e-3) == "done"
    assert "stopping" not in caplog.text and "[early stop] didn't stop early" in caplog.text
    assert (out / "t-best.safetensors").exists() and (out / "t-000003.safetensors").exists()


def test_off_without_validation(tiny, tmp_path, caplog):
    T, cfg = tiny
    caplog.set_level("INFO")
    assert _run(T, cfg, tmp_path / "noval", validation_early_stop=True, max_train_epochs=2) == "done"
    assert "[early stop] off: it needs Image Validation" in caplog.text


def test_form_and_command(tmp_path):
    from fizgig.families import form, launch, registry
    desc = registry.get("moss_image")
    rows = {r["key"]: r for sec in form.form(desc)["sections"] if sec["id"] == "validation" for r in sec["rows"]}
    assert rows["VALIDATION_EARLY_STOP"]["kind"] == "check" and form.DEFAULTS["VALIDATION_EARLY_STOP"] is False
    assert form.DEFAULTS["VALIDATION_PATIENCE"] == "3"
    st = form.state(desc, {**form.DEFAULTS})
    assert {"VALIDATION_EARLY_STOP", "VALIDATION_PATIENCE"} <= set(st["disabled"])   # Image Validation off
    st = form.state(desc, {**form.DEFAULTS, "VALIDATION_LOSS": True})
    assert "VALIDATION_EARLY_STOP" not in st["disabled"] and "VALIDATION_PATIENCE" in st["disabled"]
    st = form.state(desc, {**form.DEFAULTS, "VALIDATION_LOSS": True, "VALIDATION_EARLY_STOP": True})
    assert not {"VALIDATION_EARLY_STOP", "VALIDATION_PATIENCE"} & set(st["disabled"])

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

    def cmd(**v):
        inp = form.inputs(desc, {**form.DEFAULTS, "LORA_NAME": "t", "LORA_OUTPUT_DIR": str(tmp_path / "out"),
                                 "DATASET_CONFIG": str(tmp_path / "dataset.toml"), **v},
                          models=models, python=sys.executable, repo_dir=str(tmp_path), image_folder=str(img),
                          cache_root=str(tmp_path / "cache"), samples_dir=str(tmp_path / "out" / "sample"))
        return launch.plan(desc, inp).stages[-1].cmd
    c = cmd(VALIDATION_LOSS=True, VALIDATION_EARLY_STOP=True, VALIDATION_PATIENCE="5")
    assert "--validation_early_stop" in c and c[c.index("--validation_patience") + 1] == "5"
    assert "--validation_early_stop" not in cmd(VALIDATION_LOSS=False, VALIDATION_EARLY_STOP=True)
    assert "--validation_early_stop" not in cmd(VALIDATION_LOSS=True)


def test_server_note_and_outputs():
    from fizgig.web import server
    line = "INFO:fizgig.families.train:[early stop] Final LoRA = best validation, step 40 (epoch 4): t.safetensors\n"
    assert server.early_stop_note(line) == ("Stopped early: validation stopped improving. Final LoRA = the best "
                                            "check, step 40 (epoch 4).")
    assert server.early_stop_note("INFO:x:[early stop] best so far: 0.7 at step 4") == ""
    assert server._CKPT.match("my-lora-best").group(1) == "my-lora"   # listed with its run, not as its own
    src = open(os.path.join(ROOT, "src/fizgig/web/server.py"), encoding="utf-8").read()
    assert 'and not _run.get("early_stop"))' in src                      # a pending Pause doesn't turn it into one
    js = open(os.path.join(ROOT, "src/fizgig/web/static/app.js"), encoding="utf-8").read()
    assert "const es = data.stop;" in js and '"lg-stop"' in js
