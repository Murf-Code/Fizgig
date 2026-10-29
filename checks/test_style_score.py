"""The Style score (Chris, 29 Oct 2026): a preview against the whole training set's look, not one paired picture -
on a style set of many characters a better-styled preview could score lower Paired just for a different fur colour."""
import json

import numpy as np

from fizgig.web import gallery_likeness as gl
from fizgig.web import sample_pairs as sp


def _png(p):
    p.write_bytes(b"x")
    return p


def test_sidecar_names_the_style_folders_even_with_nothing_paired(tmp_path):
    d = json.loads(sp.sidecar_text("", ["a"], str(tmp_path), [str(tmp_path), str(tmp_path / "sub")]))
    assert d["pairs"] == [] and d["style"] == [str(tmp_path), str(tmp_path / "sub")]
    assert "style" not in json.loads(sp.sidecar_text("", ["a"], str(tmp_path)))


def test_style_images_skip_masks_and_cap(tmp_path, monkeypatch):
    for i in range(10):
        _png(tmp_path / f"{i:02d}.png")
    from fizgig.dataset.masks import MASK_SUFFIX
    _png(tmp_path / f"03{MASK_SUFFIX}.png")
    monkeypatch.setattr(sp, "STYLE_MAX_IMAGES", 4)
    got = sp.style_images([str(tmp_path)])
    assert len(got) == 4 and got[0].endswith("00.png") and got[-1].endswith("09.png")
    assert not any(MASK_SUFFIX in g for g in got)


def test_style_pass_scores_every_preview_once_and_keeps_paired(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "STYLE_TOP_K", 1)          # its closest training picture (a varied set: top-k, not mean)
    data, s = tmp_path / "data", tmp_path / "sample"
    data.mkdir(), s.mkdir()
    for n in ("a.png", "b.png"):
        _png(data / n)
    (s / ".pairs_run.json").write_text(sp.sidecar_text("", ["p"], str(data), [str(data)]))
    for n in ("run_e000001_00_x.png", "run_e000002_00_x.png"):
        _png(s / n)
    gl.merge_paired(str(s), {"run_e000001_00_x.png": {"score": 0.5, "image": "a.png", "index": 0}})
    vec = {str(data / "a.png"): np.array([1.0, 0.0]), str(data / "b.png"): np.array([0.0, 1.0]),
           str(s / "run_e000001_00_x.png"): np.array([0.8, 0.6]), str(s / "run_e000002_00_x.png"): np.array([1.0, 0.0])}
    assert sp.style_pass(str(s), style_fn=lambda p: vec[p]) == 2
    st = gl.read(str(s))["style"]
    assert st["run_e000002_00_x.png"]["raw"] > st["run_e000001_00_x.png"]["raw"]
    assert gl.read(str(s))["paired"]                       # the paired scores stay
    assert (s / ".style_run.npz").exists()                 # the training side is worked out once
    assert sp.style_pass(str(s), style_fn=lambda p: 1 / 0) == 0


def test_shown_score_bands():
    assert sp.style_show(0.775) < 0.3 and 0.55 < sp.style_show(0.878) < 0.7 and sp.style_show(0.93) >= 0.85
    assert sp.style_show(0.5) == 0 and sp.style_show(0.99) == 1


def test_launch_writes_the_style_folders_with_paired_on():
    from fizgig.families import form
    assert form.DEFAULTS["SAMPLE_STYLE"] is True
