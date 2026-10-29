"""Where Image Prep puts things, pinned: every mode that makes crops names them the same way - Crop_<word>_NNN for
what you type, FaceCrop_NNN for faces, numbered on after any file of that name already there (Chris: "no matter
what crop setting the file name should remain consistent"). Resize + close-up crops saves them beside the resized
pictures; Close-up crops only (Face Crop Only) adds the crop beside its picture, which stays in the dataset with its caption
untouched (Chris: "when it crops an image, I still want that original image to remain in the dataset") - no originals
folder for crops. Resized originals go into ONE originals folder - originals, or originals_2 when an earlier run's
originals are there.
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
def test_crop_only_other_names_crops_like_auto_prep(tmp_path, trim, earlier):
    _folder(tmp_path, earlier=earlier)
    _things(tmp_path, "Face Crop Only", _caps(trim))
    want = [f"p{i}.{e}" for i in range(4) for e in ("jpg", "txt")]                # the pictures stay, untouched
    want += [f"Crop_sheath_{i:03d}.{e}" for i in range(1, 5) for e in ("png", "txt")]
    if earlier:
        want.append("originals/old.jpg")
    assert _tree(tmp_path) == sorted(want)
    for i in range(4):
        assert (tmp_path / f"p{i}.txt").read_text() == CAP                       # the picture's caption, as it was
    assert (tmp_path / "Crop_sheath_001.txt").read_text() == ("ruaidri, sheath, wolf" if trim else CAP)
    w, h = Image.open(tmp_path / "Crop_sheath_001.png").size
    assert w == h and w < 200


def test_every_crop_mode_names_crops_the_same(tmp_path):
    """Close-up crops only and Resize + close-up crops give a picture's crop the same name and the same caption."""
    names = {}
    for mode in ("Face Crop Only", "Auto Prep (Face Crops)"):
        d = tmp_path / mode.split()[0]
        d.mkdir()
        _folder(d, n=2, earlier=False)
        _things(d, mode, _caps(True))
        names[mode] = {n: (d / n).read_bytes() for n in os.listdir(d) if n.startswith("Crop_")}
    assert sorted(names["Face Crop Only"]) == sorted(names["Auto Prep (Face Crops)"]) == [
        "Crop_sheath_001.png", "Crop_sheath_001.txt", "Crop_sheath_002.png", "Crop_sheath_002.txt"]
    assert names["Face Crop Only"]["Crop_sheath_001.txt"] == names["Auto Prep (Face Crops)"]["Crop_sheath_001.txt"]


@pytest.mark.parametrize("mode", ["Face Crop Only", "Auto Prep (Face Crops)"])
def test_numbering_goes_on_after_old_crops_and_left_behind_captions(tmp_path, mode):
    """Chris's folder: Crop_sheath_001 to 044 deleted, Crop_sheath_045.txt left behind - a new crop must not take it."""
    _folder(tmp_path, n=2, earlier=False)
    (tmp_path / "Crop_sheath_045.txt").write_text("an old crop's caption")
    Image.new("RGB", (8, 8)).save(tmp_path / "Crop_sheath_007.png")
    Image.new("L", (8, 8)).save(tmp_path / "Crop_sheath_050-masklabel.png")   # a mask left behind counts too
    _things(tmp_path, mode, _caps(False))
    names = os.listdir(tmp_path)
    assert "Crop_sheath_051.png" in names and "Crop_sheath_052.png" in names and "Crop_sheath_053.png" not in names
    assert (tmp_path / "Crop_sheath_045.txt").read_text() == "an old crop's caption"
    assert (tmp_path / "Crop_sheath_051.txt").read_text() == CAP
    if mode == "Face Crop Only":                         # an old crop is a crop already: never cropped again
        assert (tmp_path / "Crop_sheath_007.png").exists()


def test_crop_only_nothing_found_stays_whole_under_its_name(tmp_path):
    class _None:
        def masks(self, *a, **k):
            return []
    _folder(tmp_path, n=1, earlier=False)
    prep.Prep(lambda _t: None, None, prep.ThingFinder(_None(), ["sheath"]), captions=_caps(True)).run(
        "Face Crop Only", str(tmp_path), 0.05, prep.OTHER_TARGET, 20.0, False)
    assert _tree(tmp_path) == ["p0.jpg", "p0.txt"]                 # left untouched: no resize, no backup
    assert (tmp_path / "p0.txt").read_text() == CAP


def test_crop_only_replace_still_keeps_the_picture_and_its_caption(tmp_path):
    _folder(tmp_path, n=1, earlier=False)
    before = (tmp_path / "p0.jpg").read_bytes()
    prep.Prep(lambda _t: None, None, prep.ThingFinder(_Seg(), ["sheath"]), captions=_caps(False)).run(
        "Face Crop Only", str(tmp_path), 0.05, prep.OTHER_TARGET, 20.0, True)
    assert _tree(tmp_path) == ["Crop_sheath_001.png", "Crop_sheath_001.txt", "p0.jpg", "p0.txt"]
    assert (tmp_path / "p0.jpg").read_bytes() == before and (tmp_path / "p0.txt").read_text() == CAP


def test_crop_only_without_caption_settings_still_keeps_the_caption(tmp_path):
    """captions=None (the desktop's Image Prep): the crop gets its picture's caption as written, as it did when the
    crop took the picture's own name."""
    _folder(tmp_path, n=1, earlier=False)
    prep.Prep(lambda _t: None, None, prep.ThingFinder(_Seg(), ["sheath"])).run(
        "Face Crop Only", str(tmp_path), 0.05, prep.OTHER_TARGET, 20.0, False)
    assert (tmp_path / "Crop_sheath_001.txt").read_text() == CAP
    assert (tmp_path / "p0.txt").read_text() == CAP and not (tmp_path / "originals").exists()


def test_mask_follows_the_crop_name(tmp_path):
    _folder(tmp_path, n=1, earlier=False)
    Image.new("L", (300, 300), 200).save(tmp_path / "p0-masklabel.png")
    _things(tmp_path, "Face Crop Only", _caps(False))
    names = _tree(tmp_path)
    assert "Crop_sheath_001-masklabel.png" in names and "p0-masklabel.png" in names   # the picture keeps its mask
    assert not any(n.startswith("originals") for n in names)


def test_resize_only_keeps_names(tmp_path):
    _folder(tmp_path, n=2, earlier=False)
    prep.Prep(lambda _t: None, None, None, captions=_caps(True)).run(
        "Resize Only", str(tmp_path), 0.05, "Largest Face", 20.0, False)
    assert _tree(tmp_path) == ["originals/p0.jpg", "originals/p1.jpg", "p0.png", "p0.txt", "p1.png", "p1.txt"]


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


def test_face_crop_only_keeps_the_pictures_and_makes_no_originals_folder(tmp_path):
    _folder(tmp_path, n=3)
    prep.Prep(lambda _t: None, _Det([F._face_info(60, 60, 140, 140)]), captions=_caps(True)).run(
        "Face Crop Only", str(tmp_path), 0.05, "Largest Face", 20.0, False)
    dirs = sorted(d for d in os.listdir(tmp_path) if os.path.isdir(tmp_path / d))
    assert dirs == ["originals"]                                                 # only the earlier run's
    files = sorted(n for n in os.listdir(tmp_path) if os.path.isfile(tmp_path / n))
    assert files == sorted([f"FaceCrop_{i:03d}.{e}" for i in range(1, 4) for e in ("png", "txt")] +   # as Auto Prep names faces
                           [f"p{i}.{e}" for i in range(3) for e in ("jpg", "txt")])
    assert all((tmp_path / f"p{i}.txt").read_text() == CAP for i in range(3))


def test_a_second_run_never_crops_its_crops(tmp_path):
    _folder(tmp_path, n=2, earlier=False)
    p = prep.Prep(lambda _t: None, None, prep.ThingFinder(_Seg(), ["sheath"]), captions=_caps(True))
    p.run("Face Crop Only", str(tmp_path), 0.05, prep.OTHER_TARGET, 20.0, False)
    for i in range(2):
        Image.fromarray(np.random.default_rng(9 + i).integers(0, 255, (300, 300, 3), dtype=np.uint8)).save(
            tmp_path / f"q{i}.jpg")
    p.run("Face Crop Only", str(tmp_path), 0.05, prep.OTHER_TARGET, 20.0, False)
    # the pictures stay and are cropped again (p0, p1 -> 003, 004; q0, q1 -> 005, 006); the crops never are
    assert sorted(n for n in os.listdir(tmp_path) if n.endswith(".png")) == [f"Crop_sheath_{i:03d}.png" for i in range(1, 7)]
    assert not any(n.startswith("originals") for n in os.listdir(tmp_path))


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
    tags = [t.strip() for t in (tmp_path / "Crop_sheath_001.txt").read_text().split(",")]
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
    assert (tmp_path / "FaceCrop_001.txt").read_text() == "ruaidri, face, wolf"


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
