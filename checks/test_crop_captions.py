"""Image Prep crops keep their picture's caption (on by default), and "Trim tags that don't fit" (off by default)
drops the general tags the WD EVA02 tagger (the default; JTP-3 Hydra in test_jtp3_tagger.py) no longer sees on the crop: only removed, never added; the trigger word,
the first tag, character / rating / meta tags and tags it doesn't know stay; sentence captions are copied as written.
The tagger is a stand-in here (the real one is a 1.2 GB download)."""
import os
import types

import numpy as np
import pytest
from PIL import Image

from fizgig import face_utils as F
from fizgig.dataset import crop_captions as C
from fizgig.dataset import prep, wd_tagger

CATS = {"wolf": 0, "solo": 0, "full body": 0, "tail": 0, "smile": 0, "standing": 0, "blue eyes": 0, "male": 0,
        "general": 9, "sensitive": 9, "krystal (star fox)": 4, "highres": 5}


class FakeTagger:
    """wd_tagger.Tagger's face: scores(img) -> {tag: probability}, categories."""

    def __init__(self, scores):
        self.categories, self._scores, self.seen = dict(CATS), scores, []

    def scores(self, img):
        self.seen.append(img.size)
        return dict(self._scores)


def test_norm():
    assert wd_tagger.norm("Long_Hair") == "long hair" == wd_tagger.norm(" long  hair ")
    assert wd_tagger.norm("krystal \\(star fox\\)") == "krystal (star fox)"
    assert wd_tagger.norm("^_^") == "^_^"


def test_trim_keeps_what_it_should():
    cap = ("ruaidri, wolf, full_body, tail, smile, krystal \\(star fox\\), general, highres, by zaush, anthro, "
           "standing, blue eyes")
    scores = {"wolf": 0.9, "full body": 0.05, "tail": 0.34, "smile": 0.35, "standing": 0.1, "blue eyes": 0.8}
    out, dropped = wd_tagger.trim(cap, scores, CATS, ["Ruaidri"], 0.35)
    assert dropped == ["full_body", "tail", "standing"]
    assert out == "ruaidri, wolf, smile, krystal \\(star fox\\), general, highres, by zaush, anthro, blue eyes"
    # the first tag and the trigger word stay even when the tagger doesn't see them
    assert wd_tagger.trim("tail, wolf, smile", {"wolf": 0.9}, CATS, ["smile"]) == ("tail, wolf, smile", [])
    s = "A wolf with a tail, standing in a field."
    assert wd_tagger.trim(s, {}, CATS) == (s, [])


def test_preprocess_pads_white_and_goes_bgr():
    img = Image.new("RGBA", (40, 20), (255, 0, 0, 255))
    x = wd_tagger.preprocess(img, 8)
    assert x.shape == (1, 8, 8, 3) and x.dtype == np.float32
    assert x[0, 4, 4].tolist() == [0.0, 0.0, 255.0]          # red, as BGR
    assert x[0, 0, 4].tolist() == [255.0, 255.0, 255.0]      # the padding is white
    clear = wd_tagger.preprocess(Image.new("RGBA", (8, 8), (0, 0, 0, 0)), 8)
    assert (clear == 255).all()                               # see-through is white too


def test_tagger_reads_the_csv_and_runs_the_session(tmp_path):
    (tmp_path / "selected_tags.csv").write_text("tag_id,name,category,count\n1,general,9,1\n2,long_hair,0,1\n"
                                                "3,krystal_(star_fox),4,1\n", encoding="utf-8")
    calls = []

    class Sess:
        def get_inputs(self):
            return [types.SimpleNamespace(name="input", shape=[1, 16, 16, 3])]

        def run(self, _out, feed):
            calls.append(feed["input"].shape)
            return [np.array([[0.7, 0.2, 0.9]], np.float32)]
    t = wd_tagger.Tagger(str(tmp_path), session=Sess())
    assert t.categories == {"general": 9, "long hair": 0, "krystal (star fox)": 4} and t.size == 16
    sc = t.scores(Image.new("RGB", (30, 10)))
    assert calls == [(1, 16, 16, 3)] and sc["long hair"] == pytest.approx(0.2)
    t.close()
    assert t.session is None


def test_model_folder_and_download(tmp_path, monkeypatch):
    assert wd_tagger.find_model(str(tmp_path)) == ""
    got = []

    def fake(repo_id, filename, local_dir):
        got.append((repo_id, filename))
        os.makedirs(local_dir, exist_ok=True)
        open(os.path.join(local_dir, filename), "w").close()
        return os.path.join(local_dir, filename)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)
    monkeypatch.setitem(wd_tagger._fetch, "state", "")
    wd_tagger.download(str(tmp_path))
    for _ in range(100):
        if wd_tagger.fetch_state(str(tmp_path))["state"] != "running":
            break
        import time
        time.sleep(0.02)
    assert wd_tagger.fetch_state(str(tmp_path))["state"] == "done"
    assert got == [(wd_tagger.REPO, "selected_tags.csv"), (wd_tagger.REPO, "model.onnx")]
    assert wd_tagger.find_model(str(tmp_path)) == os.path.join(str(tmp_path), wd_tagger.FOLDER)


def test_find_caption_extension_order(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"")
    assert C.find_caption(str(img), ".txt") is None
    (tmp_path / "a.caption").write_text("c")
    assert C.find_caption(str(img), ".txt") == str(tmp_path / "a.caption")
    (tmp_path / "a.txt").write_text("t")
    assert C.find_caption(str(img), ".txt") == str(tmp_path / "a.txt")
    (tmp_path / "a.tags").write_text("g")
    assert C.find_caption(str(img), ".tags") == str(tmp_path / "a.tags")


class _Det:
    backend_name = "insightface"

    def __init__(self, faces):
        self.faces = faces

    def detect_from_pil(self, img):
        return list(self.faces)

    def get_largest(self, faces):
        return max(faces, key=lambda f: f.area) if faces else None


def _pic(path, seed=0):
    Image.fromarray(np.random.default_rng(seed).integers(0, 255, (160, 200, 3), dtype=np.uint8)).save(path)


CAP = "ruaidri, wolf, full body, tail, smile"
SEEN = {"wolf": 0.8, "smile": 0.6, "full body": 0.02, "tail": 0.1}


def test_auto_prep_copies_and_trims(tmp_path):
    _pic(tmp_path / "a.png", 1)
    _pic(tmp_path / "b.png", 2)
    _pic(tmp_path / "c.png", 3)
    (tmp_path / "a.txt").write_text(CAP)
    (tmp_path / "b.txt").write_text("A wolf standing in a forest. He smiles.")
    (tmp_path / "c.txt").write_text(CAP)
    (tmp_path / "FaceCrop_003.txt").write_text("mine")             # a caption already there for c's crop: kept
    log = []
    tagger = FakeTagger(SEEN)
    caps = C.CropCaptions(ext=".txt", keep=["ruaidri"], tagger=tagger)
    prep.Prep(log.append, _Det([F._face_info(20, 10, 60, 58)]), captions=caps).run(
        "Auto Prep (Face Crops)", str(tmp_path), 4.2, "Largest Face", 20.0, False)
    assert (tmp_path / "FaceCrop_001.txt").read_text() == "ruaidri, wolf, smile"
    assert (tmp_path / "FaceCrop_002.txt").read_text() == "A wolf standing in a forest. He smiles."
    assert (tmp_path / "FaceCrop_003.txt").read_text() == "mine"
    assert (tmp_path / "a.txt").read_text() == CAP                  # the picture's own caption is untouched
    assert tagger.seen == [(66, 66)]                                 # the crop is what gets tagged, not the picture
    text = "".join(log)
    assert "caption: dropped full body, tail" in text and "FaceCrop_003.txt already there, kept" in text


def test_copy_only_by_default_and_off_switch(tmp_path):
    for sub, caps in (("on", C.CropCaptions()), ("off", C.CropCaptions(copy=False)), ("none", None)):
        d = tmp_path / sub
        d.mkdir()
        _pic(d / "a.png", 4)
        (d / "a.txt").write_text(CAP)
        prep.Prep(lambda _t: None, _Det([F._face_info(20, 10, 60, 58)]), captions=caps).run(
            "Auto Prep (Face Crops)", str(d), 4.2, "Largest Face", 20.0, False)
        crop_cap = d / "FaceCrop_001.txt"
        assert crop_cap.exists() == (sub == "on")
        if sub == "on":
            assert crop_cap.read_text() == CAP                       # no tagger: copied as is


def test_crop_only_trims_in_place_and_keeps_the_original(tmp_path):
    _pic(tmp_path / "a.jpg", 5)
    (tmp_path / "a.txt").write_text(CAP)
    prep.Prep(lambda _t: None, _Det([F._face_info(20, 10, 60, 58)]),
              captions=C.CropCaptions(keep=["ruaidri"], tagger=FakeTagger(SEEN))).run(
        "Face Crop Only", str(tmp_path), 4.2, "Largest Face", 20.0, False)
    assert (tmp_path / "a.txt").read_text() == "ruaidri, wolf, smile"
    assert (tmp_path / "originals" / "a.txt").read_text() == CAP
    assert (tmp_path / "originals" / "a.jpg").exists()


def test_crop_only_no_face_leaves_the_caption(tmp_path):
    _pic(tmp_path / "a.jpg", 6)
    (tmp_path / "a.txt").write_text(CAP)
    t = FakeTagger(SEEN)
    prep.Prep(lambda _t: None, _Det([]), captions=C.CropCaptions(tagger=t)).run(
        "Face Crop Only", str(tmp_path), 4.2, "Largest Face", 20.0, False)
    assert (tmp_path / "a.txt").read_text() == CAP and t.seen == []


def test_other_crops_get_trimmed_captions(tmp_path):
    from checks.test_crop_other import FakeSeg
    _pic(tmp_path / "a.png", 7)
    (tmp_path / "a.txt").write_text(CAP)
    seg = FakeSeg({"paw": [((120, 100, 160, 140), 0.9)]})
    prep.Prep(lambda _t: None, None, prep.ThingFinder(seg, ["paw"]),
              C.CropCaptions(keep=["ruaidri"], tagger=FakeTagger({"wolf": 0.5, "tail": 0.4}))).run(
        "Auto Prep (Face Crops)", str(tmp_path), 4.2, prep.OTHER_TARGET, 20.0, False)
    assert (tmp_path / "Crop_paw_001.txt").read_text() == "ruaidri, wolf, tail"


def test_page_settings_checked(tmp_path, monkeypatch):
    from fizgig import user_data
    from fizgig.web import prep_jobs
    monkeypatch.setattr(user_data, "MODELS_DIR", str(tmp_path))
    c, folder = prep_jobs.caption_settings({"copy": True, "trim": False, "ext": "caption", "trigger": " me "})
    assert c == {"copy": True, "trim": False, "tagger": "wd-eva02", "keep_score": 0.35, "ext": ".caption",
                 "trigger": "me"} and folder == ""          # no tagger picked (an older page): the WD tagger
    assert prep_jobs.caption_settings(None) == (None, "")
    with pytest.raises(ValueError, match="press Download under it"):
        prep_jobs.caption_settings({"trim": True})
    with pytest.raises(ValueError, match="between 0 and 1"):
        prep_jobs.caption_settings({"trim": True, "keep_score": "1.5"})
    d = tmp_path / wd_tagger.FOLDER
    d.mkdir()
    for f in wd_tagger.FILES:
        (d / f).write_text("")
    c, folder = prep_jobs.caption_settings({"trim": True, "keep_score": "0.3"})
    assert folder == str(d) and c["keep_score"] == 0.3
    caps = prep_jobs.crop_captions(c, tagger="T")
    assert caps.tagger == "T" and caps.keep_score == 0.3
    assert prep_jobs.crop_captions({**c, "trim": False}, tagger="T").tagger is None
    with pytest.raises(ValueError, match="Download under it"):          # stopped before a run starts
        monkeypatch.setattr(user_data, "MODELS_DIR", str(tmp_path / "none"))
        prep_jobs.start(str(tmp_path), "Face Crop Only", 1.0, "Largest Face", 20, False, "", {"trim": True})


def test_run_frees_the_tagger(tmp_path, monkeypatch):
    import time
    from fizgig import user_data
    from fizgig.web import prep_jobs
    models = tmp_path / "models" / wd_tagger.FOLDER
    models.mkdir(parents=True)
    for f in wd_tagger.FILES:
        (models / f).write_text("")
    data = tmp_path / "data"
    data.mkdir()
    _pic(data / "a.png", 8)
    (data / "a.txt").write_text(CAP)
    monkeypatch.setattr(user_data, "MODELS_DIR", str(tmp_path / "models"))
    tagger = FakeTagger(SEEN)
    closed = []
    tagger.close = lambda: closed.append(1)
    monkeypatch.setattr(prep_jobs, "_tagger", lambda folder, name=None: tagger)
    monkeypatch.setattr(prep_jobs, "faces_available", lambda: True)
    monkeypatch.setattr(prep_jobs, "_face_detector", lambda: _Det([F._face_info(20, 10, 60, 58)]))
    prep_jobs.start(str(data), "Auto Prep (Face Crops)", 4.2, "Largest Face", 20, False, "",
                    {"copy": True, "trim": True, "keep_score": "0.35", "trigger": "ruaidri"})
    for _ in range(200):
        if prep_jobs.status()["state"] != "running":
            break
        time.sleep(0.05)
    assert prep_jobs.status()["state"] == "done", prep_jobs.status()["log"]
    assert (data / "FaceCrop_001.txt").read_text() == "ruaidri, wolf, smile" and closed == [1]
