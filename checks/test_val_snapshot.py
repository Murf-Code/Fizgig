"""Validation keeps its startup snapshot: a caption fix or auto-recaption later (the cache's embedding rewritten)
never reaches it, a resume checks the same images and captions again, and a resume with other validation settings
starts a new section of the line (logged as "seg", drawn unjoined)."""
import json, os, re, shutil, subprocess, sys
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fizgig.training.step_log import StepLog, read, loss_path  # noqa: E402
from fizgig.training.val_loss import FixedValidation  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "src", "fizgig")
KEYS = [f"img{i:02d}" for i in range(8)]


def _batch(k, cap):
    return {"item_keys": [k], "latents": torch.full((1, 2), float(KEYS.index(k))), "cond__c": torch.full((1, 3), cap)}


def _fill(v, cap):
    for k in KEYS:
        v.offer(_batch(k, cap))


def test_a_caption_change_never_reaches_the_kept_snapshot(tmp_path):
    v = FixedValidation(KEYS, images=3, picked=None)
    v.snapshot_path = str(tmp_path / "snap.pt")
    _fill(v, 1.0)
    assert v.ready and os.path.isfile(v.snapshot_path)
    _fill(v, 2.0)                                  # the next epoch trains on the new embedding
    assert all(float(b["cond__c"][0, 0]) == 1.0 for b in v.batches.values())


def test_resume_restores_the_snapshot(tmp_path):
    p = str(tmp_path / "snap.pt")
    v = FixedValidation(KEYS, images=3, seed=7)
    v.snapshot_path = p
    _fill(v, 1.0)
    r = FixedValidation(KEYS, images=3, seed=7)
    assert r.restore(p) == "kept" and r.ready and r.seg == 0
    _fill(r, 2.0)                                  # the cache now holds the fixed caption: still the old one here
    assert all(float(b["cond__c"][0, 0]) == 1.0 for b in r.batches.values())


def test_resume_with_other_settings_starts_a_new_section(tmp_path):
    p = str(tmp_path / "snap.pt")
    v = FixedValidation(KEYS, images=3, seed=7)
    v.snapshot_path = p
    _fill(v, 1.0)
    r = FixedValidation(KEYS, images=4, seed=7)    # another image picked
    assert r.restore(p) == "changed" and r.seg == 1 and not r.batches
    _fill(r, 2.0)
    r2 = FixedValidation(KEYS, images=4, seed=7)   # resumed again, same as last time: the same section
    assert r2.restore(p) == "kept" and r2.seg == 1
    assert all(float(b["cond__c"][0, 0]) == 2.0 for b in r2.batches.values())


def test_no_snapshot_is_as_before(tmp_path):
    r = FixedValidation(KEYS, images=3)
    assert r.restore(str(tmp_path / "missing.pt")) == "none" and r.seg == 0 and not r.batches
    (tmp_path / "bad.pt").write_bytes(b"not a torch file")
    assert FixedValidation(KEYS, images=3).restore(str(tmp_path / "bad.pt")) == "none"


def test_seg_in_the_log(tmp_path):
    lg = StepLog(str(tmp_path), "x")
    lg.add(1, 1, 0.1)
    lg.add_validation(1, 1, 0.12, [0.1], 1.0)
    lg.add_validation(2, 2, 0.11, [0.1], 1.0, seg=1)
    lg.flush()
    d = read(loss_path(str(tmp_path), "x"))
    assert d["val"] == [[1, 1, 0.12, [0.1]], [2, 2, 0.11, [0.1], 1]]
    assert '"seg"' not in open(loss_path(str(tmp_path), "x")).read().splitlines()[1]


def test_trainer_wiring():
    src = open(os.path.join(ROOT, "families", "train.py"), encoding="utf-8").read()
    assert '_vsnap = os.path.join(output_dir, f".fizgig_val_{output_name}.pt")' in src
    assert "_vs = validator.restore(_vsnap)" in src and "os.remove(_vsnap)" in src
    assert "seg=validator.seg)" in src
    assert 'so the validation line starts a new "' in src and 'section from epoch {start_epoch + 1}.' in src


def test_chart_never_joins_sections():
    if not shutil.which("node"):
        pytest.skip("node not installed")
    app = open(os.path.join(ROOT, "web", "static", "app.js"), encoding="utf-8").read()
    fn = re.search(r"function valSections\(val\) \{.*?\n\}\n", app, re.S).group(0)
    assert 'for (const part of valSections(val)) line("lg-val", part, 2);' in app
    val = [[1, 1, 0.12, []], [2, 2, 0.11, []], [3, 3, 0.13, [], 1], [4, 4, 0.12, [], 1]]
    js = fn + f"console.log(JSON.stringify(valSections({json.dumps(val)}).map((p) => p.map((x) => x[0]))));"
    assert json.loads(subprocess.check_output(["node", "-e", js], text=True)) == [[1, 2], [3, 4]]
    js = fn + "console.log(JSON.stringify(valSections([[1,1,0.1,[]],[2,2,0.1,[]]]).length));"
    assert json.loads(subprocess.check_output(["node", "-e", js], text=True)) == 1
