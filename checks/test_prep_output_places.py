"""Where Image Prep puts things, pinned: Auto Prep with Other… saves Crop_<word>_NNN close-ups beside the resized
pictures; Face Crop Only turns each picture into its crop under its own name. Either way a run's originals (and the
captions they had) go into ONE originals folder - originals, or originals_2 when an earlier run's originals are there.
Chris's Zaush run (Face Crop Only, Other… sheath, captions on) put each original in its own folder, originals_2 to
originals_46: the caption step forgot the run's originals folder after every picture, so the next one picked a new one."""
import os

import numpy as np
import pytest
from PIL import Image

from fizgig import face_utils as F
from fizgig.dataset import crop_captions as C
from fizgig.dataset import prep

from test_crop_captions import FakeTagger as _FakeTagger, _Det


class FakeTagger(_FakeTagger):
    """test_crop_captions' stand-in, which also knows the targets (a tag it doesn't know always stays, so a target
    would never be trimmed and the checks below would prove nothing)."""

    def __init__(self, scores):
        super().__init__(scores)
        self.categories.update({"sheath": 0, "face": 0, "paw pads": 0, "pawpads": 0})

CAP = "ruaidri, sheath, wolf, full body"
SEEN = {"sheath": 0.9, "wolf": 0.8, "full body": 0.02}


class _Seg:
    def masks(self, image, prompts, threshold=0.5, refine=2, max_each=None):
        return [{"prompt": p, "score": 0.9, "box": [60.0, 60.0, 140.0, 140.0], "mask": None} for p in prompts]


def _folder(tmp_path, n=4, earlier=True):
    if earlier:                                        # an earlier run's originals already there (Chris's folder)
        os.makedirs(tmp_path / "originals")
        Image.new("RGB", (8, 8)).save(tmp_path / "originals" / "old.jpg")
    for i in range(n):
        Image.fromarray(np.random.default_rng(i).integers(0, 255, (300, 300, 3), dtype=np.uint8)).save(
            tmp_path / f"p{i}.jpg")
        (tmp_path / f"p{i}.txt").write_text(CAP)


def _tree(d):
    return sorted(os.path.relpath(os.path.join(r, f), d).replace(os.sep, "/") for r, _, fs in os.walk(d) for f in fs)


def _caps(trim):
    return C.CropCaptions(keep=["ruaidri"], tagger=FakeTagger(SEEN) if trim else None)


def _things(tmp_path, mode, caps):
    prep.Prep(lambda _t: None, None, prep.ThingFinder(_Seg(), ["sheath"]), captions=caps).run(
        mode, str(tmp_path), 0.05, prep.OTHER_TARGET, 20.0, False)


@pytest.mark.parametrize("trim", [False, True])
@pytest.mark.parametrize("earlier", [False, True])
def test_crop_only_other_one_originals_folder_names_kept(tmp_path, trim, earlier):
    _folder(tmp_path, earlier=earlier)
    _things(tmp_path, "Face Crop Only", _caps(trim))
    o = "originals_2" if earlier else "originals"
    want = [f"{o}/p{i}.{e}" for i in range(4) for e in ("jpg", "txt")] + [f"p{i}.{e}" for i in range(4) for e in ("png", "txt")]
    if earlier:
        want.append("originals/old.jpg")
    if not trim:                       # nothing trimmed: the caption isn't changed, so no copy of it is kept
        want = [w for w in want if not (w.startswith(o + "/") and w.endswith(".txt"))]
    assert _tree(tmp_path) == sorted(want)
    assert not any(n.startswith("Crop_") for n in os.listdir(tmp_path))
    w, h = Image.open(tmp_path / "p0.png").size                      # the picture is its crop, under its own name
    assert w == h and w < 200


@pytest.mark.parametrize("earlier", [False, True])
def test_auto_prep_other_crop_names_beside_the_pictures(tmp_path, earlier):
    _folder(tmp_path, earlier=earlier)
    _things(tmp_path, "Auto Prep (Face Crops)", _caps(True))
    o = "originals_2" if earlier else "originals"
    names = _tree(tmp_path)
    for i in range(1, 5):
        assert f"Crop_sheath_{i:03d}.png" in names and f"Crop_sheath_{i:03d}.txt" in names, names
    for i in range(4):
        assert f"p{i}.png" in names and f"{o}/p{i}.jpg" in names, names
    assert not any(n.startswith("originals_3") for n in names) and not any("/Crop_" in n for n in names), names


def test_face_crop_only_with_captions_one_originals_folder(tmp_path):
    _folder(tmp_path, n=3)
    prep.Prep(lambda _t: None, _Det([F._face_info(60, 60, 140, 140)]), captions=_caps(True)).run(
        "Face Crop Only", str(tmp_path), 0.05, "Largest Face", 20.0, False)
    dirs = sorted(d for d in os.listdir(tmp_path) if os.path.isdir(tmp_path / d))
    assert dirs == ["originals", "originals_2"]
    assert sorted(os.listdir(tmp_path / "originals_2")) == [f"p{i}.{e}" for i in range(3) for e in ("jpg", "txt")]


def test_a_second_run_uses_a_fresh_folder(tmp_path):
    _folder(tmp_path, n=2, earlier=False)
    p = prep.Prep(lambda _t: None, None, prep.ThingFinder(_Seg(), ["sheath"]), captions=_caps(True))
    p.run("Face Crop Only", str(tmp_path), 0.05, prep.OTHER_TARGET, 20.0, False)
    for i in range(2):
        Image.fromarray(np.random.default_rng(9 + i).integers(0, 255, (300, 300, 3), dtype=np.uint8)).save(
            tmp_path / f"q{i}.jpg")
    p.run("Face Crop Only", str(tmp_path), 0.05, prep.OTHER_TARGET, 20.0, False)
    assert {"q0.jpg", "q1.jpg"} <= set(os.listdir(tmp_path / "originals_2"))   # the second run's own folder
    assert sorted(os.listdir(tmp_path / "originals")) == [f"p{i}.{e}" for i in range(2) for e in ("jpg", "txt")]
    assert not (tmp_path / "originals_3").exists()


# ---- the crop's target stays in its caption ----
# Chris's sheath run: on 23 of the 45 crops the trim dropped "sheath" itself - JTP-3 scored it 0.00 to 0.06 on those
# small (58-191 px) crops although each one is a sheath.
LOW = {"sheath": 0.0, "wolf": 0.8, "full body": 0.02}


@pytest.mark.parametrize("cap", ["ruaidri, Sheath, wolf, full body", "ruaidri, wolf, sheath, full body"])
def test_trim_keeps_the_find_this_word(tmp_path, cap):
    _folder(tmp_path, n=1, earlier=False)
    (tmp_path / "p0.txt").write_text(cap)
    caps = C.CropCaptions(keep=["ruaidri"], tagger=FakeTagger(LOW), closeup=False)
    _things(tmp_path, "Face Crop Only", caps)
    tags = [t.strip() for t in (tmp_path / "p0.txt").read_text().split(",")]
    assert "full body" not in tags and "wolf" in tags
    assert any(t.lower() == "sheath" for t in tags), tags


def test_target_match_ignores_case_spaces_underscores():
    caps = C.CropCaptions(tagger=FakeTagger({"wolf": 0.8}), closeup=False)
    caps.targets = ["paw pads"]
    for cap, kept in (("x, Paw_Pads, wolf, tail", "Paw_Pads"), ("x, pawpads, wolf", "pawpads"),
                      ("x, paw-pads, wolf", "paw-pads")):
        text, dropped, added = caps.caption_for(cap, Image.new("RGB", (64, 64)))
        assert kept in text.split(", ") and kept not in dropped and not added, (cap, text)


def test_face_is_the_target_of_face_crops(tmp_path):
    _folder(tmp_path, n=1, earlier=False)
    (tmp_path / "p0.txt").write_text("ruaidri, face, wolf, full body")
    caps = C.CropCaptions(keep=["ruaidri"], tagger=FakeTagger({"wolf": 0.8, "face": 0.0}), closeup=False)
    prep.Prep(lambda _t: None, _Det([F._face_info(60, 60, 140, 140)]), captions=caps).run(
        "Face Crop Only", str(tmp_path), 0.05, "Largest Face", 20.0, False)
    assert (tmp_path / "p0.txt").read_text() == "ruaidri, face, wolf"


@pytest.mark.parametrize("add", [False, True])
def test_missing_target_added_only_with_add_missing_tags(tmp_path, add):
    _folder(tmp_path, n=1, earlier=False)
    (tmp_path / "p0.txt").write_text("ruaidri, wolf, full body")
    caps = C.CropCaptions(keep=["ruaidri"], tagger=FakeTagger(LOW), closeup=True, add_score=0.9 if add else None)
    _things(tmp_path, "Auto Prep (Face Crops)", caps)
    crop = (tmp_path / "Crop_sheath_001.txt").read_text()
    assert crop == ("ruaidri, wolf, sheath, close-up" if add else "ruaidri, wolf, close-up")
    assert (tmp_path / "p0.txt").read_text() == "ruaidri, wolf, full body"     # the resized picture: as written


def test_added_target_follows_underscore_style():
    caps = C.CropCaptions(tagger=FakeTagger({"wolf": 0.8}), closeup=False, add_score=0.9)
    caps.targets = ["paw pads"]
    text, _d, added = caps.caption_for("x, long_hair, wolf", Image.new("RGB", (64, 64)))
    assert added == ["paw_pads"] and text.endswith(", paw_pads")
    caps.add_score = None
    assert caps.caption_for("x, long_hair, wolf", Image.new("RGB", (64, 64)))[2] == []
