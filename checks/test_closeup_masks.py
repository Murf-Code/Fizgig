"""Make masks on close-ups (face crops filling the frame) with the character finder: small pictures enlarged smoothly
(INTER_AREA enlarged them in pixel blocks and the finder then saw nothing), the head finder's box when the character
finder still misses, the whole character as the head when the head finder misses, and a note saying what happened
or why nothing was found. No models here: stand-ins answer. Real people's masks: test_furry_masks.py."""
import os
import types

import cv2
import numpy as np
import pytest
from PIL import Image

from checks.test_furry_masks import FakeChars, FakeHeads, FakeSegmenter, _folder
from fizgig.dataset import automask, charmask

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


class Sess:
    def __init__(self):
        self.x = None

    def get_inputs(self):
        return [types.SimpleNamespace(name="img")]

    def run(self, _out, feed):
        self.x = feed["img"]
        return [np.zeros((1, 1, charmask.SIDE, charmask.SIDE), np.float32)]


def test_small_pictures_are_enlarged_smoothly():
    s = Sess()
    f = charmask.CharacterFinder("unused", session=s)
    a = np.zeros((8, 8, 3), np.uint8)
    a[:, 4:] = 255                                          # black | white, 8 px: enlarged 128 times
    f.mask(Image.fromarray(a))
    row = s.x[0, 0, charmask.SIDE // 2]
    mid = ((row > 0.05) & (row < 0.95)).sum()
    assert mid > 20                                         # a soft edge, not one 128-pixel block next to another
    assert s.x.min() >= 0.0 and s.x.max() <= 1.0            # Lanczos overshoot clipped


def test_big_pictures_shrink_as_before():
    s = Sess()
    f = charmask.CharacterFinder("unused", session=s)
    a = np.random.default_rng(1).integers(0, 255, (1500, 2000, 3), dtype=np.uint8)
    f.mask(Image.fromarray(a))
    want = cv2.resize(a.astype(np.float32) / 255.0, (1024, 768), interpolation=cv2.INTER_AREA)
    top = (1024 - 768) // 2
    assert np.array_equal(s.x[0].transpose(1, 2, 0)[top:top + 768], want)


def test_closeup_uses_the_head_box():
    char = np.zeros((40, 40), np.float32)                   # the character finder saw nothing
    asked = []

    def heads():
        asked.append(1)
        return [(4, 2, 36, 38)]
    m, note, used = charmask.drawn_mask(char, "person", heads)
    assert m is not None and m.mean() > 0.9 and "head finder's box" in note and used.mean() > 0.9
    m, note, _ = charmask.drawn_mask(char, "person", lambda: [])
    assert m is None and note == "no character found (the head finder found no head either)"
    m, note, _ = charmask.drawn_mask(char, "person")
    assert m is None and note == "no character found"
    # a character found: the head finder isn't asked for the whole character, and nothing changes
    char[:, 10:30] = 1.0
    asked.clear()
    m, note, used = charmask.drawn_mask(char, "person", heads)
    assert not asked and note == "" and np.array_equal(m, charmask.mask_from(char, "person")) and used is char


def test_a_finder_that_saw_the_head_keeps_its_own_outline():
    char = np.zeros((100, 100), np.float32)
    char[10:30, 10:25] = 1.0                                # a small character (3%), its head seen
    m, note, _ = charmask.drawn_mask(char, "person", lambda: [(10, 10, 25, 16)])
    assert note == "" and m.sum() == char.sum()


def test_head_kinds_on_closeups():
    full = np.zeros((40, 40), np.float32)
    full[:, 2:38] = 1.0                                     # a character filling the frame, no head found
    m, note, _ = charmask.drawn_mask(full, "head", lambda: [])
    assert m is not None and m.sum() == full.sum() and "whole character counted as the head" in note
    m, note, _ = charmask.drawn_mask(full, "body_clothes", lambda: [])
    assert m is None and note == "only a head here, nothing else to mask"
    small = np.zeros((40, 40), np.float32)
    small[:, 10:20] = 1.0
    m, note, _ = charmask.drawn_mask(small, "head", lambda: [])
    assert m is None and note == "a character but no head found"
    # with a head: as before
    hr = charmask.head_region((40, 40), [(12, 0, 18, 8)])
    m, note, _ = charmask.drawn_mask(small, "head", lambda: [(12, 0, 18, 8)])
    assert note == "" and np.array_equal(m, charmask.mask_from(small, "head", hr))
    m, note, _ = charmask.drawn_mask(full, "clothes", lambda: [])
    assert m is None and "can't tell clothes from fur" in note
    m, note, _ = charmask.drawn_mask(np.ones((40, 40), np.float32), "background", lambda: [])
    assert m is None and "no background" in note


def test_auto_closeup_photos_go_to_the_person_finder():
    h, w = 80, 100
    nothing = np.zeros((h, w), np.float32)
    assert charmask.looks_like_photo(FakeSegmenter().classes(Image.new("RGB", (w, h))), nothing)
    assert not charmask.looks_like_photo(FakeSegmenter(face=False).classes(Image.new("RGB", (w, h))), nothing)


def test_folder_closeup_gets_the_head_box(tmp_path):
    d = _folder(tmp_path, 1)
    log = []
    c = automask.make_folder(str(d), "person", str(tmp_path), who="drawn", characters=FakeChars(0.0, 0.0),
                             heads=FakeHeads([(10, 10, 60, 60)]), log=log.append)
    assert c["made"] == 1 and "(a close-up: the head finder's box used)" in "".join(log)
    m = np.asarray(Image.open(d / "c0-masklabel.png")) > 0
    assert m[20:50, 20:50].all() and not m[:, 80:].any()
    d2 = tmp_path / "none"
    d2.mkdir()
    Image.open(d / "c0.png").save(d2 / "c0.png")
    log = []
    c = automask.make_folder(str(d2), "person", str(tmp_path), who="drawn", characters=FakeChars(0.0, 0.0),
                             heads=FakeHeads([]), log=log.append)
    assert c["no_person"] == 1 and "no character found (the head finder found no head either)" in "".join(log)


def test_preview_says_why(tmp_path, monkeypatch):
    from fizgig import user_data
    from fizgig.web import prep_jobs
    d = _folder(tmp_path, 2)
    monkeypatch.setattr(user_data, "MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(prep_jobs, "quick_check", lambda kind, who: "")
    monkeypatch.setattr(charmask, "CharacterFinder", lambda path: FakeChars(0.0, 0.0))
    monkeypatch.setattr(charmask, "find_model", lambda m: "x")
    monkeypatch.setattr(prep_jobs, "_head_finder", lambda: FakeHeads([]))
    shots, _ = prep_jobs.quick_preview(str(d), "person", "drawn")
    assert len(shots) == 2 and all(s[2] == 0 and s[4].startswith("no character found") for s in shots)
    monkeypatch.setattr(prep_jobs, "_head_finder", lambda: FakeHeads([(10, 10, 60, 60)]))
    shots, _ = prep_jobs.quick_preview(str(d), "person", "drawn", shown=())
    assert all(s[2] > 0.3 and "head finder's box" in s[4] for s in shots)
    def broken():
        raise ValueError("no YOLO here")
    monkeypatch.setattr(prep_jobs, "_head_finder", broken)   # a close-up fallback that can't run: no mask, as before
    shots, _ = prep_jobs.quick_preview(str(d), "person", "drawn")
    assert all(s[2] == 0 and s[4] == "no character found (the head finder found no head either)" for s in shots)
    with pytest.raises(ValueError, match="no YOLO"):
        prep_jobs.quick_preview(str(d), "head", "drawn")     # the head kinds still need it up front


def test_page_shows_the_note():
    js = open(os.path.join(ROOT, "src", "fizgig", "web", "static", "app.js"), encoding="utf-8").read()
    server = open(os.path.join(ROOT, "src", "fizgig", "web", "server.py"), encoding="utf-8").read()
    assert '"note": note' in server and "for name, pic, cover, found_by, note in shots" in server
    assert '${p.note || "nothing found"}' in js and "p.cover && p.note ?" in js
