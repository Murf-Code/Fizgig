"""Image Prep's Crop to: the YOLO face model's other classes (Fdetailer v1.1 names face, penis, pussy, anus, sheath,
pawpads). No real model here (its weights are AGPL and never shipped): a stand-in onnxruntime session with Fdetailer's
metadata and output shapes (a YOLO11-seg: output0 (1, 4+6+32, N), output1 the mask protos). Checks the class names are
read from the model, the decode picks boxes by class, Crop to offers only the faces and Other… (no choice per class),
Other… sends a phrase that names a class to the YOLO model and the rest to SAM 3.1 (so SAM isn't needed when YOLO
finds it all), and an older build's saved "YOLO class: <name>" becomes Other… finding <name>."""
import importlib.machinery
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
NAMES6 = "{0: 'face', 1: 'penis', 2: 'pussy', 3: 'anus', 4: 'sheath', 5: 'pawpads'}"


def _table(rows, nc=6, nm=32, n=64):
    """(1, 4+nc+nm, n): rows = [(cx, cy, w, h, cls, score)] in the 640 letterbox, the rest zero."""
    t = np.zeros((4 + nc + nm, n), np.float32)
    for i, (cx, cy, w, h, c, s) in enumerate(rows):
        t[:4, i] = (cx, cy, w, h)
        t[4 + c, i] = s
        t[4 + nc:, i] = 0.7                       # mask coefficients: never read as class scores
    return t[None]


class FakeSess:
    """onnxruntime.InferenceSession as Fdetailer looks: input images (1, 3, 640, 640), names in the metadata."""

    def __init__(self, names=NAMES6, rows=(), nc=6):
        self.names, self.rows, self.nc, self.runs = names, list(rows), nc, 0

    def get_inputs(self):
        return [types.SimpleNamespace(name="images", shape=[1, 3, 640, 640])]

    def get_modelmeta(self):
        return types.SimpleNamespace(custom_metadata_map={"names": self.names, "imgsz": "[640, 640]"})

    def run(self, _o, feeds):
        self.runs += 1
        assert feeds["images"].shape == (1, 3, 640, 640)
        return [_table(self.rows, self.nc), np.zeros((1, 32, 160, 160), np.float32)]


@pytest.fixture
def fake_ort(monkeypatch):
    """A stand-in onnxruntime module; returns a holder whose .sess the next InferenceSession becomes."""
    pytest.importorskip("cv2")                      # NMS is OpenCV's (not in checks/requirements.txt)
    holder = types.SimpleNamespace(sess=FakeSess())
    mod = types.ModuleType("onnxruntime")
    mod.__spec__ = importlib.machinery.ModuleSpec("onnxruntime", None)
    mod.SessionOptions = lambda: types.SimpleNamespace(log_severity_level=0)
    mod.InferenceSession = lambda path, sess_options=None, providers=None: holder.sess
    monkeypatch.setitem(sys.modules, "onnxruntime", mod)
    monkeypatch.setattr(F.YoloOnnxBackend, "_sessions", {})
    return holder


def _model(tmp_path):
    p = tmp_path / "fdetailer_v11.onnx"
    p.write_bytes(b"x")
    return str(p)


# ---- reading the classes from the model ----
def test_classes_read_from_the_model(fake_ort, tmp_path):
    y = F.YoloOnnxBackend(_model(tmp_path))
    y.load()
    assert y.names == {0: "face", 1: "penis", 2: "pussy", 3: "anus", 4: "sheath", 5: "pawpads"}
    assert y.class_ids == [0]                                         # faces: class 0, as before
    assert y.extra_classes() == {1: "penis", 2: "pussy", 3: "anus", 4: "sheath", 5: "pawpads"}
    assert [F.class_label(n) for n in y.extra_classes().values()] == ["Penis", "Pussy", "Anus", "Sheath", "Paw pads"]
    assert F.class_label("left_hand") == "Left hand"
    assert F.class_key("Paw pads") == F.class_key("pawpads") == F.class_key("paw-pads") == "pawpads"


def test_face_only_model_offers_just_faces(fake_ort, tmp_path):
    from fizgig.web import prep_jobs
    fake_ort.sess = FakeSess("{0: 'head'}", nc=1)                   # the free anime head finder's one class
    y = F.YoloOnnxBackend(_model(tmp_path))
    y.load()
    assert y.extra_classes() == {} and prep_jobs.yolo_classes(y) == []
    assert prep_jobs.yolo_words(["head", "paw"], y) == {"head": 0}


def test_yolo_backend_only_with_the_yolo_finder(fake_ort, tmp_path):
    m = _model(tmp_path)
    assert F.yolo_backend({"face_detector": "insightface", "face_yolo_model": m}) is None
    assert F.yolo_backend({"face_detector": "auto", "face_yolo_model": m}) is None
    b = F.yolo_backend({"face_detector": "yolo", "face_yolo_model": m, "face_yolo_conf": 0.4})
    assert b is not None and b.conf == 0.4 and 5 in b.extra_classes()
    assert F.yolo_backend({"face_detector": "yolo", "face_yolo_model": str(tmp_path / "gone.onnx")}) is None


# ---- the decode: boxes picked by class ----
def test_decode_picks_by_class_and_faces_are_unchanged(fake_ort, tmp_path):
    rgb = np.zeros((480, 640, 3), np.uint8)                            # letterbox: r = 1, top = 80
    fake_ort.sess = FakeSess(rows=[(100, 180, 40, 40, 0, 0.9),          # a face
                                   (300, 280, 80, 120, 1, 0.8),         # a penis
                                   (302, 281, 80, 118, 1, 0.6),         # the same penis again: NMS keeps one
                                   (500, 400, 30, 20, 5, 0.7),          # paw pads
                                   (50, 50, 10, 10, 2, 0.2)])           # under the face finder's 0.35: dropped
    y = F.YoloOnnxBackend(_model(tmp_path), conf=0.35)
    faces = y.detect_rgb(rgb)
    assert len(faces) == 1 and np.allclose(faces[0].bbox, [80, 80, 120, 120], atol=1)
    assert faces[0].gender == "unknown" and faces[0].source == "yolo"
    got = y.find_classes(rgb, [1, 5, 2])
    assert [(c, round(s, 2)) for *_b, s, c in got] == [(1, 0.8), (5, 0.7)]
    x1, y1, x2, y2 = got[0][:4]
    assert np.allclose((x1, y1, x2, y2), (260, 140, 340, 260), atol=1)   # back in the picture's own pixels
    assert y.find_classes(rgb, [3]) == []


# ---- Other…: a phrase naming a class goes to YOLO, the rest to SAM 3.1 ----
class FakeSeg:
    def __init__(self, hits):
        self.hits, self.calls = hits, []

    def masks(self, image, prompts, threshold=0.5, refine=2, max_each=None):
        self.calls.append(list(prompts))
        return [{"prompt": p, "score": s, "box": list(map(float, b)), "mask": None}
                for p in prompts for b, s in self.hits.get(p, []) if s > threshold]


class FakeYolo:
    label = "fdetailer_v11.onnx"
    names = {0: "face", 1: "penis", 2: "pussy", 3: "anus", 4: "sheath", 5: "pawpads"}

    def __init__(self, boxes):
        self.boxes, self.calls = boxes, []

    def all_classes(self):
        return dict(self.names)

    def extra_classes(self):
        return {k: v for k, v in self.names.items() if k}

    def find_classes(self, rgb, ids):
        self.calls.append(list(ids))
        return [b for b in self.boxes if b[5] in ids]


def test_routing_yolo_for_classes_sam_for_the_rest():
    from fizgig.web import prep_jobs
    y = FakeYolo([(120, 40, 180, 120, 0.8, 1), (20, 20, 40, 44, 0.6, 1), (10, 100, 40, 130, 0.7, 5)])
    assert prep_jobs.yolo_words(["Penis", "paw pads", "tail", "face"], y) == {"Penis": 1, "paw pads": 5, "face": 0}
    seg = FakeSeg({"tail": [((150, 120, 190, 158), 0.9)]})
    f = prep.ThingFinder(seg, ["tail", "Penis", "paw pads"], yolo=y, yolo_words=prep_jobs.yolo_words(
        ["tail", "Penis", "paw pads"], y))
    hits = f.find(Image.new("RGB", (200, 160)))
    assert seg.calls == [["tail"]] and y.calls == [[1, 5]]
    assert [(h.gender, h.source) for h in hits] == [("tail", "sam3"), ("Penis", "yolo"), ("Penis", "yolo"),
                                                   ("paw pads", "yolo")]
    picked = f.per_word(hits)
    assert [p.gender for p in picked] == ["tail", "Penis", "paw pads"]
    assert picked[1].bbox == (120, 40, 180, 120)                       # the biggest penis
    assert "YOLO" in f.describe() and "SAM 3.1, finding tail" in f.describe()
    only = prep.ThingFinder(None, ["penis"], yolo=y, yolo_words={"penis": 1})   # no SAM at all
    assert [h.gender for h in only.find(Image.new("RGB", (200, 160)))] == ["penis", "penis"]
    assert only.describe() == "fdetailer_v11.onnx (YOLO), finding penis"


def test_find_words_needs_sam_only_for_what_yolo_cant_find(tmp_path, monkeypatch):
    from fizgig import user_data
    from fizgig.dataset import sammask
    from fizgig.web import prep_jobs
    monkeypatch.setattr(user_data, "MODELS_DIR", str(tmp_path))
    y = FakeYolo([])
    assert prep_jobs.find_words(prep.OTHER_TARGET, "penis, Paw pads", y) == (["penis", "Paw pads"], "")
    with pytest.raises(ValueError, match="needs SAM 3.1"):
        prep_jobs.find_words(prep.OTHER_TARGET, "penis, tail", y)
    with pytest.raises(ValueError, match="needs SAM 3.1"):                 # not the YOLO finder: SAM, as before
        prep_jobs.find_words(prep.OTHER_TARGET, "penis", types.SimpleNamespace(
            all_classes=lambda: {}))
    (tmp_path / sammask.NAME).write_bytes(b"x")
    assert prep_jobs.find_words(prep.OTHER_TARGET, "penis, tail", y) == (["penis", "tail"], str(tmp_path / sammask.NAME))
    # an older build's Crop to "YOLO class: <name>": Other… finding <name>
    assert prep.class_of("YOLO class: penis") == "penis" and prep.class_of("Largest Face") == ""
    assert prep.migrate_target("YOLO class: pawpads", "tail") == (prep.OTHER_TARGET, "pawpads")
    assert prep.migrate_target("Largest Face", "paw") == ("Largest Face", "paw")
    assert prep.migrate_target(prep.OTHER_TARGET, "paw") == (prep.OTHER_TARGET, "paw")
    assert prep_jobs.find_words("YOLO class: pawpads", "ignored", y) == (["pawpads"], "")
    (tmp_path / sammask.NAME).unlink()
    monkeypatch.setattr(prep_jobs, "_yolo", lambda: None)             # not the YOLO finder now: SAM, like Other…
    with pytest.raises(ValueError, match="needs SAM 3.1"):
        prep_jobs.find_words("YOLO class: penis", "")
    assert prep_jobs.find_words("Largest Face", "paw") == ([], "")        # face choices untouched


def _pic(path, w=200, h=160, seed=0):
    Image.fromarray(np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)).save(path)


def test_class_choice_runs_without_sam(tmp_path, monkeypatch):
    """Other… with Find this = penis (here sent as an older build's "YOLO class: penis"), Auto Prep: Crop_penis_001.png from each picture with one, SAM 3.1 never loaded; Face Crop
    Only: the biggest one, a picture without one left whole."""
    from fizgig.dataset import sammask
    from fizgig.web import prep_jobs
    data = tmp_path / "data"
    data.mkdir()
    _pic(data / "a.png", 200, 160, 1)
    y = FakeYolo([(100, 40, 140, 60, 0.9, 1), (10, 10, 20, 26, 0.95, 1)])
    monkeypatch.setattr(prep_jobs, "_yolo", lambda: y)
    monkeypatch.setattr(prep_jobs, "faces_available", lambda: True)
    monkeypatch.setattr(sammask, "segmenter", lambda *a, **k: (_ for _ in ()).throw(AssertionError("SAM loaded")))
    prep_jobs.start(str(data), "Auto Prep (Face Crops)", 4.2, "YOLO class: penis", 20, False, "")
    for _ in range(200):
        if prep_jobs.status()["state"] != "running":
            break
        time.sleep(0.05)
    st = prep_jobs.status()
    assert st["state"] == "done", st["log"]
    assert (data / "Crop_penis_001.png").exists() and not (data / "Crop_penis_002.png").exists()  # 10x16: too small
    assert "fdetailer_v11.onnx (YOLO), finding penis" in st["log"] and "Close-ups made: 1" in st["log"]
    assert not sammask._use.locked()
    log = []
    _pic(tmp_path / "b.png", 200, 160, 2)
    prep.Prep(log.append, None, prep.ThingFinder(None, ["penis"], yolo=y, yolo_words={"penis": 1})).run(
        "Face Crop Only", str(tmp_path), 4.2, prep.OTHER_TARGET, 0.0, False)
    assert Image.open(tmp_path / "Crop_penis_001.png").size == (40, 40) and "[penis]" in "".join(log)
    y.boxes = []
    _pic(tmp_path / "b.png", 200, 160, 2)
    log = []
    prep.Prep(log.append, None, prep.ThingFinder(None, ["penis"], yolo=y, yolo_words={"penis": 1})).run(
        "Face Crop Only", str(tmp_path), 4.2, prep.OTHER_TARGET, 0.0, False)
    assert Image.open(tmp_path / "b.png").size == (200, 160) and "Nothing found in b.png" in "".join(log)


def test_test_button_boxes_and_labels(tmp_path, monkeypatch):
    from fizgig.web import prep_jobs
    _pic(tmp_path / "a.png", 200, 160, 3)
    y = FakeYolo([(100, 40, 160, 100, 0.83, 1), (10, 100, 40, 130, 0.71, 5)])
    monkeypatch.setattr(prep_jobs, "_yolo", lambda: y)
    assert prep_jobs.find_words("YOLO class: penis", "") == (["penis"], "")    # an older saved choice
    words, sam = prep_jobs.find_words(prep.OTHER_TARGET, "penis")
    r = prep_jobs.test_things(str(tmp_path / "a.png"), words, sam)
    assert r["things"] and r["image"].startswith("data:image/jpeg;base64,")
    assert r["faces"] == [{"gender": "penis", "area": 3600, "selected": True, "source": "yolo", "score": 0.83}]
    r = prep_jobs.test_things(str(tmp_path / "a.png"), ["penis", "paw pads"], "")
    assert [(f["gender"], f["score"], f["selected"]) for f in r["faces"]] == [("penis", 0.83, True),
                                                                              ("paw pads", 0.71, True)]
    y.boxes = []
    assert prep_jobs.test_things(str(tmp_path / "a.png"), ["penis"], "")["faces"] == []


def test_summary_reads_right(tmp_path):
    _pic(tmp_path / "a.png", 200, 160, 1)
    s = prep.summary(str(tmp_path), "Face Crop Only", False, 1.0, None, F.class_label("penis").lower())
    assert "a close-up of the penis in each picture" in s["lines"][0] and "Crop_penis_001.png" in s["lines"][0]
    s = prep.summary(str(tmp_path), "Auto Prep (Face Crops)", False, 1.0, None, F.class_label("pawpads").lower())
    assert "a close-up of each paw pads found" in s["lines"][0]
    server = open(os.path.join(HERE, "..", "src", "fizgig", "web", "server.py"), encoding="utf-8").read()
    assert 'face, find = _prep.migrate_target(g("face"), g("find"))' in server
    assert "find = \", \".join(sammask_words(find)) if _prep.is_other(face) else \"\"" in server


def test_face_settings_lists_the_classes(monkeypatch, tmp_path):
    from fizgig.web import prep_jobs
    monkeypatch.setattr(prep_jobs, "_yolo", lambda: FakeYolo([]))
    monkeypatch.setattr(F, "PREFS_FILE", str(tmp_path / "prefs.json"))
    f = prep_jobs.face_settings()
    assert [c["label"] for c in f["classes"]] == ["Penis", "Pussy", "Anus", "Sheath", "Paw pads"]
    assert f["classes"][4] == {"label": "Paw pads", "name": "pawpads"} and "face" in f["yolo_names"]
    monkeypatch.setattr(prep_jobs, "_yolo", lambda: None)
    f = prep_jobs.face_settings()
    assert f["classes"] == [] and f["yolo_names"] == []


# ---- the page ----
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
  fn("prepFaceOptions") + fn("prepMigrate") +
  fn("prepTrimBlocked") + fn("prepFindBlocked") + "function prepBlocked() { return prepFindBlocked() || prepTrimBlocked(); }\n" +
  fn("prepFindShow") + fn("prepTaggerShow") + fn("prepTaggerInfo") + src.match(/const prepTaggerName = [^\n]*\n/)[0] +
  src.match(/const TAGGER_KEEP = [^\n]*\n/)[0] + src.match(/const TAGGER_HINTS = [\s\S]*?\n\};\n/)[0];
const FS = {classes: [["penis", "Penis"], ["pussy", "Pussy"], ["anus", "Anus"], ["sheath", "Sheath"], ["pawpads", "Paw pads"]]
  .map(([n, l]) => ({label: l, name: n})),
  yolo_names: ["face", "penis", "pussy", "anus", "sheath", "pawpads"]};
const S = {prep: {opts: {faces: ["Largest Face", "Largest Male Face", "Largest Female Face"], other: "Other…", face_tools: true},
                   running: false, face: FS, maskOpts: {sam: {path: ""}, tagger: {path: "", gb: 1.2, fetch: {}}}}};
const api = async () => ({}); const maskInfo = () => {};
eval(code + "; global.prepFindShow = prepFindShow; global.prepBlocked = prepBlocked; global.prepFaceOptions = prepFaceOptions; global.prepMigrate = prepMigrate;");
const $ = (id) => document.getElementById(id);
document.querySelector('input[name="prepMode"][value="Face Crop Only"]').checked = true;
const look = () => ({hint: $("prepFindHint").textContent, block: prepBlocked(), test: $("prepTest").disabled,
  testLabel: $("prepTest").textContent, testTitle: $("prepTest").title, row: $("prepFindRow").style.display !== "none"});
const out = {};
prepFaceOptions(FS);
out.menu = [...$("prepFace").options].map((o) => [o.value, o.text]);
// an older build's saved "YOLO class: ..." choice
out.mig = prepMigrate({face: "YOLO class: pawpads", find: "", mode: "Face Crop Only"}, FS);
out.migNoModel = prepMigrate({face: "YOLO class: penis", find: "tail"}, {classes: []});
out.untouched = prepMigrate({face: "Largest Female Face", find: "paw"}, FS);
prepFaceOptions(FS, out.mig.face); $("prepFind").value = out.mig.find; prepFindShow();
out.migShown = {face: $("prepFace").value, find: $("prepFind").value, ...look()};
prepFaceOptions(FS, "YOLO class: penis"); out.fallback = $("prepFace").value;   // never offered now
prepFaceOptions(FS, "Largest Female Face"); out.oldSaved = $("prepFace").value;
prepFaceOptions(FS, "Other…");
$("prepFind").value = "Penis, paw pads"; prepFindShow(); out.allYolo = look();
$("prepFind").value = "penis, tail"; prepFindShow(); out.mixedNoSam = look();
S.prep.maskOpts.sam.path = "/m/sam.safetensors"; prepFindShow(); out.mixed = look();
S.prep.face = {classes: [], yolo_names: []}; S.prep.maskOpts.sam.path = "";
$("prepFind").value = "penis"; prepFindShow(); out.notYolo = look();
console.log(JSON.stringify(out));
"""


def test_page_menu_and_other_routing():
    from checks.test_val_picker import JSDOM_DIRS
    if not shutil.which("node"):
        pytest.skip("node not installed")
    d = next((p for p in JSDOM_DIRS if os.path.isdir(os.path.join(p, "jsdom"))), None)
    if d is None:
        pytest.skip("jsdom not installed")
    out = json.loads(subprocess.check_output(["node", "-e", JS], text=True,
                                             env={**os.environ, "NODE_PATH": d, "STATIC": os.path.abspath(STATIC)}))
    assert out["menu"] == [[f, f] for f in ["Largest Face", "Largest Male Face", "Largest Female Face", "Other…"]]
    assert out["mig"] == {"face": "Other…", "find": "paw pads", "mode": "Face Crop Only", "migrated": True}
    assert out["migNoModel"]["face"] == "Other…" and out["migNoModel"]["find"] == "penis"
    assert out["untouched"] == {"face": "Largest Female Face", "find": "paw"}
    m = out["migShown"]
    assert m["face"] == "Other…" and m["find"] == "paw pads" and m["row"] and m["block"] == "" and not m["test"]
    assert m["testLabel"] == "Test finding this on one photo…" and "SAM 3.1 isn't needed" in m["hint"]
    assert out["fallback"] == "Largest Face"                  # "YOLO class: ..." isn't a choice any more
    assert out["oldSaved"] == "Largest Female Face"           # old saved values unchanged
    a = out["allYolo"]
    assert a["block"] == "" and not a["test"] and "SAM 3.1 isn't needed" in a["hint"]
    m = out["mixedNoSam"]
    assert m["test"] and m["testTitle"].startswith("Finding by words needs SAM 3.1") and "needs SAM 3.1" in m["block"]
    assert out["mixed"]["block"] == "" and "finds penis; SAM 3.1 finds the rest" in out["mixed"]["hint"]
    assert "needs SAM 3.1" in out["notYolo"]["block"]          # not the YOLO finder: SAM, as before


def test_page_wiring():
    js = open(os.path.join(STATIC, "app.js"), encoding="utf-8").read()
    assert "saved = prepMigrate(S.prepSaved || {}, o.face_settings)" in js
    assert "prepFaceOptions(o.face_settings, saved.face)" in js
    assert "if (saved.migrated) savePrepSoon();" in js         # the moved choice is saved
    assert "prepClass" not in js and "(f || {}).classes || []).find" in js   # no per-class menu choices
