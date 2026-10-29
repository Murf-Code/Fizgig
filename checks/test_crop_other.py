"""Image Prep's Face dropdown, Other…: crop to what the Find this box names, found by SAM 3.1. No model here: a stand-in
segmenter answers with boxes (the real SAM 3.1 needs its 1.75 GB file and runs on Chris's GPU). The face choices must
run exactly as before: the face paths are compared with the code as it was (99d8ab02's prep.py)."""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import types

import numpy as np
import pytest
from PIL import Image

from fizgig import face_utils as F
from fizgig.dataset import prep

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "..", "src", "fizgig", "web", "static")
ROOT = os.path.join(HERE, "..")


class FakeSeg:
    """SAM3Segmenter.masks() as a table: prompt -> [(box, score)]; records the prompts and arguments it was asked."""

    def __init__(self, hits):
        self.hits, self.calls = hits, []

    def masks(self, image, prompts, threshold=0.5, refine=2, max_each=None):
        self.calls.append((list(prompts), threshold, refine, image.shape))
        out = []
        for p in prompts:
            for box, score in self.hits.get(p, []):
                if score > threshold:
                    out.append({"prompt": p, "score": score, "box": [float(v) for v in box], "mask": None})
        return out


def _pic(path, w=200, h=160, seed=0):
    Image.fromarray(np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)).save(path)


# ---- finding ----
def test_each_phrase_searched_separately_and_cutoff_kept():
    seg = FakeSeg({"paw": [((10, 10, 50, 40), 0.9), ((60, 60, 70, 65), 0.95)], "tail": [((0, 0, 30, 30), 0.4)]})
    f = prep.ThingFinder(seg, ["paw", "tail", "paw"])
    hits = f.find(Image.new("RGB", (200, 160)))
    assert seg.calls[0][0] == ["paw", "tail"] and seg.calls[0][1] == prep.FIND_SCORE == 0.5
    assert seg.calls[0][2] == 0                               # boxes only: no mask refining
    assert [h.bbox for h in hits] == [(10, 10, 50, 40)]        # 10x5 is under the minimum side; 0.4 under the cutoff
    assert hits[0].gender == "paw" and hits[0].source == "sam3"


def test_boxes_clamped_to_the_picture():
    seg = FakeSeg({"tail": [((-30.4, -5, 250.6, 90.2), 0.8)]})
    (h,) = prep.ThingFinder(seg, ["tail"]).find(Image.new("RGB", (200, 160)))
    assert h.bbox == (0, 0, 200, 90) and h.area == 200 * 90


def test_per_word_picks_the_biggest_of_each_and_skips_the_same_box():
    seg = FakeSeg({"paw": [((0, 0, 40, 40), 0.9), ((100, 100, 180, 150), 0.6)],
                   "hand": [((101, 101, 180, 150), 0.9)], "tail": [((0, 100, 60, 160), 0.7)]})
    f = prep.ThingFinder(seg, ["paw", "hand", "tail"])
    hits = f.find(Image.new("RGB", (200, 160)))
    picked = f.per_word(hits)
    assert [(p.gender, p.bbox) for p in picked] == [("paw", (100, 100, 180, 150)), ("tail", (0, 100, 60, 160))]
    assert f.largest(hits).bbox == (100, 100, 180, 150)
    assert f.per_word([]) == [] and f.largest([]) is None


def test_word_slug():
    assert prep.word_slug("the left paw") == "the_left_paw"
    assert prep.word_slug("  ../\\ ") == "thing" and len(prep.word_slug("x" * 80)) == 24


def test_is_other():
    assert prep.is_other(prep.OTHER_TARGET) and prep.is_other("Other")
    assert not any(prep.is_other(t) for t in prep.FACE_TARGETS)


# ---- cropping ----
def test_auto_prep_crops_each_word_with_padding_square_and_names(tmp_path):
    _pic(tmp_path / "a.png", 200, 160, 1)
    _pic(tmp_path / "b.png", 200, 160, 2)
    Image.new("RGB", (8, 8)).save(tmp_path / "Crop_paw_004.png")          # numbering carries on; never cropped again
    seg = FakeSeg({"paw": [((100, 40, 140, 60), 0.9)], "tail": [((150, 120, 190, 158), 0.8)]})
    log = []
    prep.Prep(log.append, None, prep.ThingFinder(seg, ["paw", "tail"])).run(
        "Auto Prep (Face Crops)", str(tmp_path), 4.2, prep.OTHER_TARGET, 20.0, False)
    text = "".join(log)
    names = sorted(os.listdir(tmp_path))
    for n in ("Crop_paw_005.png", "Crop_paw_006.png", "Crop_tail_001.png", "Crop_tail_002.png", "a.png", "b.png"):
        assert n in names, (n, names)
    assert "Skipped (derivative): Crop_paw_004.png" in text and "Close-ups made: 4" in text
    src = Image.open(tmp_path / "originals" / "a.png") if (tmp_path / "originals" / "a.png").exists() else None
    pic = np.random.default_rng(1).integers(0, 255, (160, 200, 3), dtype=np.uint8)
    want = F.crop_to_face(Image.fromarray(pic), F._face_info(100, 40, 140, 60), 20.0)   # 40x20 + 20% -> 56x28 -> 56 square
    assert want.size == (56, 56)
    assert np.array_equal(np.asarray(Image.open(tmp_path / "Crop_paw_005.png")), np.asarray(want))
    tail = Image.open(tmp_path / "Crop_tail_001.png")
    assert tail.size[0] <= 200 and tail.size[1] <= 160              # clamped at the picture's edge
    assert src is None                                               # PNGs at size: nothing to keep safe


def test_crop_only_nothing_found_left_whole(tmp_path):
    _pic(tmp_path / "a.png", 200, 160, 3)
    _pic(tmp_path / "b.jpg", 200, 160, 4)
    seg = FakeSeg({"paw": []})
    log = []
    prep.Prep(log.append, None, prep.ThingFinder(seg, ["paw"])).run(
        "Face Crop Only", str(tmp_path), 4.2, prep.OTHER_TARGET, 20.0, False)
    text = "".join(log)
    assert "Nothing found in a.png, not cropped" in text and "Crops: 0 | Nothing found: 2" in text
    assert "Left as is (nothing to crop): b.jpg" in text
    assert sorted(os.listdir(tmp_path)) == ["a.png", "b.jpg"]                  # untouched, nothing in originals


def test_crop_only_crops_the_biggest(tmp_path):
    _pic(tmp_path / "a.png", 200, 160, 5)
    seg = FakeSeg({"paw": [((10, 10, 30, 30), 0.9)], "hand": [((50, 50, 110, 100), 0.7)]})
    log = []
    prep.Prep(log.append, None, prep.ThingFinder(seg, ["paw", "hand"])).run(
        "Face Crop Only", str(tmp_path), 4.2, prep.OTHER_TARGET, 0.0, False)
    assert Image.open(tmp_path / "Crop_hand_001.png").size == (60, 60)       # named for the word that was found
    assert (tmp_path / "a.png").exists() and not (tmp_path / "originals").exists()   # the picture stays
    assert "[hand]" in "".join(log)


# ---- the face paths, as they were ----
def _old_prep():
    src = subprocess.check_output(["git", "-C", ROOT, "show", "99d8ab02:src/fizgig/dataset/prep.py"], text=True)
    mod = types.ModuleType("prep_old")
    exec(compile(src, "prep_old.py", "exec"), mod.__dict__)
    return mod


class _Det:
    backend_name = "insightface"

    def __init__(self, faces):
        self.faces = faces

    def detect_from_pil(self, img):
        return list(self.faces)

    def get_largest(self, faces):
        return max(faces, key=lambda f: f.area) if faces else None

    def get_largest_by_gender(self, faces, gender, fallback_to_any=True):
        m = [f for f in faces if f.gender == gender]
        return max(m, key=lambda f: f.area) if m else (self.get_largest(faces) if fallback_to_any else None)


@pytest.mark.parametrize("mode", ["Auto Prep (Face Crops)", "Face Crop Only"])
@pytest.mark.parametrize("label", prep.FACE_TARGETS)
def test_face_paths_unchanged(tmp_path, mode, label):
    try:
        old = _old_prep()
    except Exception:
        pytest.skip("git history not here")
    faces = [F._face_info(20, 10, 60, 58, "female", 0.8), F._face_info(100, 20, 170, 100, "male", 0.9)]
    out = {}
    for name, mod in (("old", old), ("new", prep)):
        d = tmp_path / name
        d.mkdir()
        _pic(d / "a.jpg", 200, 160, 7)
        _pic(d / "b.png", 200, 160, 8)
        (d / "a.txt").write_text("x")
        log = []
        mod.Prep(log.append, _Det(faces)).run(mode, str(d), 1.0, label, 20.0, False)
        files = {}
        for root, _, fs in os.walk(d):
            for f in fs:
                p = os.path.join(root, f)
                files[os.path.relpath(p, d).replace(os.sep, "/")] = open(p, "rb").read()
        out[name] = ("".join(log).replace(str(d), "D"), files)
    if mode == "Auto Prep (Face Crops)":
        assert out["old"] == out["new"]
        return
    # Close-up crops only now names its crops as Auto Prep does (FaceCrop_NNN, the caption with it) and adds them
    # beside the pictures, which stay with their captions untouched: the same crops, pixel for pixel
    old_f, new_f = out["old"][1], out["new"][1]
    assert sorted(new_f) == ["FaceCrop_001.png", "FaceCrop_001.txt", "FaceCrop_002.png", "a.jpg", "a.txt", "b.png"]
    assert new_f["FaceCrop_001.png"] == old_f["a.png"] and new_f["FaceCrop_002.png"] == old_f["b.png"]
    assert new_f["FaceCrop_001.txt"] == new_f["a.txt"] == b"x"
    assert new_f["a.jpg"] == old_f["originals/a.jpg"]


# ---- the run: checks before it starts, SAM freed after ----
def test_find_words_checks(tmp_path, monkeypatch):
    from fizgig import user_data
    from fizgig.dataset import sammask
    from fizgig.web import prep_jobs
    monkeypatch.setattr(user_data, "MODELS_DIR", str(tmp_path))
    assert prep_jobs.find_words("Largest Face", "paw") == ([], "")
    with pytest.raises(ValueError, match="Find this"):
        prep_jobs.find_words(prep.OTHER_TARGET, " , ")
    with pytest.raises(ValueError, match="Download on the Masks card"):
        prep_jobs.find_words(prep.OTHER_TARGET, "paw")
    (tmp_path / sammask.NAME).write_bytes(b"x")
    assert prep_jobs.find_words(prep.OTHER_TARGET, "paw, tail") == (["paw", "tail"], str(tmp_path / sammask.NAME))
    with pytest.raises(ValueError, match="Find this"):              # stopped before any job starts
        prep_jobs.start(str(tmp_path), "Face Crop Only", 1.0, prep.OTHER_TARGET, 20, False, "")


def test_run_uses_sam_and_frees_it(tmp_path, monkeypatch):
    from fizgig import user_data
    from fizgig.dataset import sammask
    from fizgig.web import prep_jobs
    models = tmp_path / "models"
    models.mkdir()
    (models / sammask.NAME).write_bytes(b"x")
    data = tmp_path / "data"
    data.mkdir()
    _pic(data / "a.png", 200, 160, 9)
    (data / "a.txt").write_text("me, tail, smile")
    monkeypatch.setattr(user_data, "MODELS_DIR", str(models))
    seg = FakeSeg({"tail": [((100, 60, 180, 140), 0.9)]})
    freed = []
    monkeypatch.setattr(sammask, "segmenter", lambda path, log=print: seg)
    monkeypatch.setattr(sammask, "unload", lambda: freed.append(1))
    monkeypatch.setattr(prep_jobs, "faces_available", lambda: False)     # SAM alone: no face tools needed
    prep_jobs.start(str(data), "Auto Prep (Face Crops)", 4.2, prep.OTHER_TARGET, 20, False, "tail",
                    {"copy": True, "trim": False, "ext": ".txt", "trigger": "me"})
    for _ in range(200):
        if prep_jobs.status()["state"] != "running":
            break
        time.sleep(0.05)
    st = prep_jobs.status()
    assert st["state"] == "done", st["log"]
    assert (data / "Crop_tail_001.png").exists() and freed == [1]
    assert (data / "Crop_tail_001.txt").read_text() == "me, tail, smile, close-up"     # the caption came along
    assert not sammask._use.locked()


# ---- the page ----
def test_page_wiring():
    html = open(os.path.join(STATIC, "index.html"), encoding="utf-8").read()
    js = open(os.path.join(STATIC, "app.js"), encoding="utf-8").read()
    server = open(os.path.join(ROOT, "src", "fizgig", "web", "server.py"), encoding="utf-8").read()
    assert 'id="prepFind"' in html and "Find this:" in html and 'placeholder="e.g. hat, tail, hand"' in html
    assert 'id="prepFindRow"' in html and 'class="field-row prep-opts prep-face" id="prepFindRow"' in html
    assert 'id="prepCapCopy" checked' in html and 'id="prepCapTrim">' in html and 'id="prepKeepScore"' in html
    assert 'find: $("prepFind").value' in js and '$("prepFind").value = saved.find' in js     # saved and loaded
    assert "capCopy: $(\"prepCapCopy\").checked" in js and "saved.capCopy !== false" in js
    assert 'if (o.other) $("prepFace").append' in js
    assert '"other": _prep.OTHER_TARGET' in server
    assert 'str(b.get("find") or ""), b.get("captions")' in server
    assert "prep_jobs.test_things(" in server
    assert '"/api/prep/tagger_fetch"' in server and '"tagger": _tagger_info()' in server
    assert "keep_score: p.keepScore" in js and 'saved.keepScore' in js


JS = r"""
const {JSDOM} = require("jsdom"); const fs = require("fs");
const html = fs.readFileSync(process.env.STATIC + "/index.html", "utf8");
const card = html.match(/<div class="section-card">\s*<div class="card-title">1 · What to do[\s\S]*?<div class="section-card">\s*<div class="card-title">2 ·/)[0];
const dom = new JSDOM("<body>" + card + "<button id='prepRun'></button><button id='prepTest'></button><div id='prepState'></div></div></body>");
global.window = dom.window; global.document = dom.window.document;
const src = fs.readFileSync(process.env.STATIC + "/app.js", "utf8");
const fn = (name) => { const i = src.indexOf("function " + name + "("); const a = src.lastIndexOf("\n", i) + 1;
  return src.slice(a, src.indexOf("\n}\n", i) + 3); };
const code = src.match(/const \$ = \(id\)[^\n]*\n/)[0] + src.match(/const el = \(tag[\s\S]*?\n};\n/)[0] +
  src.match(/const prepSettings = [\s\S]*?\}\);\n/)[0] + src.match(/const prepOther = [^\n]*\n/)[0] +
  src.match(/const classKey = [^\n]*\n/)[0] + fn("prepYoloWords") +
  fn("prepTrimBlocked") + fn("prepFindBlocked") + "function prepBlocked() { return prepFindBlocked() || prepTrimBlocked(); }\n" +
  fn("prepFindShow") + fn("prepTaggerShow") + fn("prepTaggerInfo") + src.match(/const prepTaggerName = [^\n]*\n/)[0] +
  src.match(/const TAGGER_KEEP = [^\n]*\n/)[0] + src.match(/const TAGGER_HINTS = [\s\S]*?\n\};\n/)[0];
const S = {prep: {opts: {other: "Other…", face_tools: true}, running: false,
                   maskOpts: {sam: {path: ""}, tagger: {path: "", gb: 1.2, fetch: {}}}}};
const api = async () => ({}); const maskInfo = () => {};
eval(code + "; global.prepFindShow = prepFindShow; global.prepBlocked = prepBlocked; global.prepSettings = prepSettings;");
const $ = (id) => document.getElementById(id);
for (const f of ["Largest Face", "Other…"]) $("prepFace").append(new window.Option(f, f));
document.querySelector('input[name="prepMode"][value="Face Crop Only"]').checked = true;
const look = () => ({row: $("prepFindRow").style.display !== "none", hint: $("prepFindHint").textContent,
  run: $("prepRun").disabled, block: prepBlocked(), finder: $("prepFinderNow").style.display !== "none",
  runTitle: $("prepRun").title, test: $("prepTest").disabled, testLabel: $("prepTest").textContent,
  testTitle: $("prepTest").title});
const out = {};
$("prepFace").value = "Largest Face"; prepFindShow(); out.face = look();
$("prepFace").value = "Other…"; prepFindShow(); out.empty = look();
$("prepFind").value = "paw, tail"; prepFindShow(); out.nosam = look();
S.prep.maskOpts.sam.path = "/m/sam.safetensors"; prepFindShow(); out.ready = look();
out.settings = prepSettings();
$("prepCapTrim").checked = true; prepFindShow();
out.trimNoTagger = {run: $("prepRun").disabled, note: $("prepTrimNote").textContent, box: $("prepTrimBox").style.display,
  btn: $("prepTaggerState").textContent};
S.prep.maskOpts.tagger.path = "/m/wd"; $("prepKeepScore").value = "2"; prepFindShow();
out.trimBadScore = {run: $("prepRun").disabled, note: $("prepTrimNote").textContent};
$("prepKeepScore").value = "0.3"; prepFindShow();
out.trimReady = {run: $("prepRun").disabled, note: $("prepTrimNote").textContent, keep: prepSettings().keepScore};
$("prepCapTrim").checked = false;
$("prepFace").value = "Largest Face"; S.prep.opts.face_tools = false; prepFindShow(); out.noFaceTools = look();
$("prepFace").value = "Other…"; prepFindShow(); out.otherNoFaceTools = look();
S.prep.opts.face_tools = true;
document.querySelector('input[name="prepMode"][value="Resize Only"]').checked = true; prepFindShow(); out.resize = look();
console.log(JSON.stringify(out));
"""


def test_page_find_box():
    from checks.test_val_picker import JSDOM_DIRS
    if not shutil.which("node"):
        pytest.skip("node not installed")
    d = next((p for p in JSDOM_DIRS if os.path.isdir(os.path.join(p, "jsdom"))), None)
    if d is None:
        pytest.skip("jsdom not installed")
    out = json.loads(subprocess.check_output(["node", "-e", JS], text=True,
                                             env={**os.environ, "NODE_PATH": d, "STATIC": os.path.abspath(STATIC)}))
    assert not out["face"]["row"] and out["face"]["finder"] and not out["face"]["run"] and out["face"]["block"] == ""
    assert out["empty"]["row"] and "Type what to find" in out["empty"]["hint"] and out["empty"]["run"]
    assert not out["empty"]["finder"]
    assert "Download on the Masks card" in out["nosam"]["hint"] and out["nosam"]["run"]
    assert out["ready"]["block"] == "" and not out["ready"]["run"] and "SAM 3.1 finds" in out["ready"]["hint"]
    assert out["settings"]["face"] == "Other…" and out["settings"]["find"] == "paw, tail"
    assert out["settings"]["capCopy"] is True and out["settings"]["capTrim"] is False     # the defaults
    assert out["settings"]["keepScore"] == "0.35"
    t = out["trimNoTagger"]
    assert t["run"] and "Download under Trim" in t["note"] and t["box"] == "" and "Download the WD tagger (1.2 GB)" in t["btn"]
    assert out["trimBadScore"]["run"] and "between 0 and 1" in out["trimBadScore"]["note"]
    assert not out["trimReady"]["run"] and out["trimReady"]["note"] == "" and out["trimReady"]["keep"] == "0.3"


def test_page_test_button_words_and_hover():
    """The Test button's words follow Crop to; greyed out, it (and Prepare) says why on hover; pressable, no hover."""
    from checks.test_val_picker import JSDOM_DIRS
    if not shutil.which("node"):
        pytest.skip("node not installed")
    d = next((p for p in JSDOM_DIRS if os.path.isdir(os.path.join(p, "jsdom"))), None)
    if d is None:
        pytest.skip("jsdom not installed")
    out = json.loads(subprocess.check_output(["node", "-e", JS], text=True,
                                             env={**os.environ, "NODE_PATH": d, "STATIC": os.path.abspath(STATIC)}))
    face, empty, nosam, ready = out["face"], out["empty"], out["nosam"], out["ready"]
    assert face["testLabel"] == "Test the face crop on one photo…" and not face["test"] and face["testTitle"] == ""
    assert face["runTitle"] == ""
    for o in (empty, nosam, ready):
        assert o["testLabel"] == "Test finding this on one photo…"
    assert empty["test"] and empty["testTitle"] == "Type what to find in Find this (e.g. hat, tail, hand)."
    assert empty["runTitle"] == empty["testTitle"]
    assert nosam["test"] and nosam["testTitle"] == nosam["runTitle"] == \
        "Finding by words needs SAM 3.1: press Download on the Masks card below first."
    assert not ready["test"] and ready["testTitle"] == "" and not ready["run"] and ready["runTitle"] == ""
    assert "Download under Trim" in out["trimNoTagger"]["note"]          # Prepare's hover is the trim note too
    nf = out["noFaceTools"]
    assert nf["test"] and nf["testTitle"].startswith("Face detection isn't installed on the Fizgig computer")
    assert nf["testLabel"] == "Test the face crop on one photo…"
    assert not out["otherNoFaceTools"]["test"]                          # Other… uses SAM 3.1, not the face tools
    r = out["resize"]
    assert r["test"] and r["testTitle"] == "Resize only makes no crops, so there's nothing to test."
    assert r["testLabel"] == "Test the crop on one photo…" and not r["run"] and r["runTitle"] == ""


def test_test_button_one_source_of_truth():
    html = open(os.path.join(STATIC, "index.html"), encoding="utf-8").read()
    js = open(os.path.join(STATIC, "app.js"), encoding="utf-8").read()
    assert 'id="prepTest">Test the crop on one photo…</button>' in html and "Test face detection" not in html
    assert js.count('$("prepTest").disabled =') == 1                   # only prepFindShow sets it
    assert '$("prepRun").title = S.prep.running ? "" : prepBlocked();' in js


def test_things_names_the_phrase(tmp_path, monkeypatch):
    """Test with Other…: each find is named by its phrase (the page shows f.gender), never 'unknown'."""
    from fizgig.dataset import sammask
    from fizgig.web import prep_jobs
    _pic(tmp_path / "a.png", 200, 160, 3)
    seg = FakeSeg({"paw": [((10, 10, 60, 50), 0.9)], "tail": [((100, 60, 180, 140), 0.8)]})
    monkeypatch.setattr(sammask, "segmenter", lambda path, log=print: seg)
    r = prep_jobs.test_things(str(tmp_path / "a.png"), ["paw", "tail"], "/m/sam")
    assert sorted(f["gender"] for f in r["faces"]) == ["paw", "tail"] and all(f["selected"] for f in r["faces"])


def test_summary_unchanged_for_faces_and_says_what_is_found(tmp_path):
    try:
        old = _old_prep()
    except Exception:
        pytest.skip("git history not here")
    _pic(tmp_path / "a.png", 200, 160, 1)
    for mode in prep.MODES:
        for rep in (False, True):
            new_s, old_s = prep.summary(str(tmp_path), mode, rep, 1.0, 2.0), old.summary(str(tmp_path), mode, rep, 1.0, 2.0)
            if mode == "Face Crop Only":                  # it now says what the crops are named; the rest as it was
                assert "FaceCrop_001.png" in new_s["lines"][0] and new_s["lines"][1:] == old_s["lines"][1:]
                assert "stay as they are" in new_s["lines"][0]
                assert new_s["warn"] == old_s["warn"]
            elif mode == "Auto Prep (Face Crops)":        # it now says the resized pictures stay beside the crops
                assert "stay in the dataset" in new_s["lines"][0] and new_s["lines"][1:] == old_s["lines"][1:]
                assert new_s["warn"] == old_s["warn"]
            else:
                assert new_s == old_s
    s = prep.summary(str(tmp_path), "Auto Prep (Face Crops)", False, 1.0, None, "paw, tail")
    assert "a close-up of each paw, tail found" in s["lines"][0] and "nothing is found are left whole" in s["lines"][-1]
