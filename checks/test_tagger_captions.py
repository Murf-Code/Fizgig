"""The Captions tab can caption with the two taggers Image Prep's Trim tags uses (WD EVA02 Large v3 and JTP-3 Hydra):
tag captions, the surest tag first, from a min score (WD 0.60 as in TagGUI, JTP-3 0.30 on its demo's scale), at most
max tags, with spaces or underscores, a rating tag only when asked for, an optional lower score for character tags,
and the trigger word in front (never twice). The same files and downloads as Trim; the worker frees a tagger when its
job ends; Florence-2 and Qwen3-VL are unchanged. The taggers are fakes here (the real ones are 1 GB downloads)."""
import io
import json
import os
import shutil
import subprocess
import sys

import pytest
from PIL import Image

from checks.test_jtp3_tagger import FakeNet, LABELS, _folder as _jtp3_folder
from fizgig.dataset import jtp3_tagger as J
from fizgig.dataset import tag_captions as T
from fizgig.dataset import wd_tagger as W
from fizgig.scripts import batch_caption as B
from fizgig.web import captions as C

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "..", "src", "fizgig", "web", "static")

WD_CSV = "tag_id,name,category,count\n" + "\n".join(f"{i},{n},{c},0" for i, (n, c) in enumerate([
    ("general", 9), ("sensitive", 9), ("explicit", 9), ("1girl", 0), ("long_hair", 0), ("smile", 0), ("^_^", 0),
    ("fox_girl", 0), ("hatsune_miku", 4), ("ohwx", 0), ("solo", 0)])) + "\n"
WD_PROBS = [0.70, 0.25, 0.05, 0.97, 0.88, 0.61, 0.65, 0.30, 0.45, 0.80, 0.59]


class FakeSession:
    def __init__(self, probs):
        import numpy as np
        self.probs = np.array([probs], dtype=np.float32)

    def get_inputs(self):
        class I:
            name, shape = "input", [1, 448, 448, 3]
        return [I()]

    def run(self, _out, feed):
        assert feed["input"].shape == (1, 448, 448, 3)
        return [self.probs]


def _wd_folder(models):
    d = os.path.join(str(models), W.FOLDER)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "selected_tags.csv"), "w").write(WD_CSV)
    open(os.path.join(d, "model.onnx"), "w").close()
    return d


def _wd(tmp_path, probs=WD_PROBS):
    return W.Tagger(_wd_folder(tmp_path), session=FakeSession(probs))


def _jtp3(tmp_path, probs):
    return J.Tagger(_jtp3_folder(tmp_path), net=FakeNet(probs), labels=LABELS)


IMG = Image.new("RGB", (64, 48), (200, 100, 50))


def _cap(tagger, tid, trigger="", **kw):
    return T.caption(T.scored_tags(tagger, IMG), tid, T.settings(kw, tid), trigger)


def test_names_defaults_and_settings():
    assert T.MODELS == {"WD EVA02 Large v3 tagger (anime tags)": "wd-eva02",
                        "JTP-3 Hydra tagger (furry / e621 tags)": "jtp3"}
    assert T.MIN_SCORE == {"wd-eva02": 0.60, "jtp3": 0.30} and T.MAX_TAGS == 255
    assert T.settings({}, "wd-eva02") == {"min_score": 0.60, "max_tags": 255, "style": "spaces", "ratings": False,
                                          "char_score": None}
    assert T.settings({"min_score": "", "char_score": " "}, "jtp3")["min_score"] == 0.30
    assert T.settings({"min_score": "0.5", "max_tags": "10", "style": "underscores", "ratings": True,
                       "char_score": "0.2"}, "jtp3") == {"min_score": 0.5, "max_tags": 10, "style": "underscores",
                                                         "ratings": True, "char_score": 0.2}
    assert T.settings({"style": "weird"}, "jtp3")["style"] == "spaces"
    for bad in ({"min_score": "2"}, {"min_score": "x"}, {"max_tags": "0"}, {"char_score": "-1"}):
        with pytest.raises(ValueError):
            T.settings(bad, "wd-eva02")
    assert T.tagger_id("WD EVA02 Large v3 tagger (anime tags)") == "wd-eva02" and T.tagger_id(C.QWEN_MODEL) == ""


def test_wd_threshold_sort_and_style(tmp_path):
    t = _wd(tmp_path)
    # 0.60 and up, surest first; the rating left out; kaomoji keep their underscore
    assert _cap(t, "wd-eva02") == "1girl, long hair, ohwx, ^_^, smile"
    assert _cap(t, "wd-eva02", style="underscores") == "1girl, long_hair, ohwx, ^_^, smile"
    assert _cap(t, "wd-eva02", min_score="0.9") == "1girl"
    assert _cap(t, "wd-eva02", min_score="0.5") == "1girl, long hair, ohwx, ^_^, smile, solo"


def test_wd_max_tags_ratings_characters_and_trigger(tmp_path):
    t = _wd(tmp_path)
    assert _cap(t, "wd-eva02", max_tags=2) == "1girl, long hair"
    # only the likeliest rating, in its place by score
    assert _cap(t, "wd-eva02", ratings=True) == "1girl, long hair, ohwx, general, ^_^, smile"
    assert _cap(t, "wd-eva02", char_score="0.4") == "1girl, long hair, ohwx, ^_^, smile, hatsune miku"
    assert _cap(t, "wd-eva02", char_score="0.5") == "1girl, long hair, ohwx, ^_^, smile"
    # a tag that is the trigger word isn't written twice (the worker puts the trigger in front)
    assert _cap(t, "wd-eva02", trigger="OHWX") == "1girl, long hair, ^_^, smile"
    assert _cap(t, "wd-eva02", trigger="ohwx", max_tags=3) == "1girl, long hair, ^_^"


def test_jtp3_scale_implications_ratings_and_characters(tmp_path):
    # raw p -> confidence 2p - 1: fox 0.9 -> 0.8, tail 0.7 -> 0.4, solo 0.6 -> 0.2, krystal 0.62 -> 0.24
    t = _jtp3(tmp_path, {"fox": 0.9, "tail": 0.7, "solo": 0.6, "krystal_(star_fox)": 0.62, "safe": 0.95,
                         "explicit": 0.55, "long_hair": 0.8})
    # canine comes with fox (implied: its score; a tie keeps the model order), hair with long hair; no rating;
    # brackets as they are
    assert _cap(t, "jtp3") == "fox, canine, hair, long hair, tail"
    assert _cap(t, "jtp3", ratings=True) == "safe, fox, canine, hair, long hair, tail"
    assert _cap(t, "jtp3", char_score="0.2") == "fox, canine, hair, long hair, tail, krystal (star fox)"
    assert _cap(t, "jtp3", char_score="0.2", style="underscores") == \
        "fox, canine, hair, long_hair, tail, krystal_(star_fox)"
    assert _cap(t, "jtp3", min_score="0.1", max_tags=6) == "fox, canine, hair, long hair, tail, krystal (star fox)"


def test_worker_writes_tag_captions_and_frees_the_tagger(tmp_path, monkeypatch):
    imgs = tmp_path / "set"
    imgs.mkdir()
    for n in ("a.png", "b.jpg"):
        IMG.save(imgs / n)
    lst = tmp_path / "list.txt"
    lst.write_text(f"{imgs / 'a.png'}\n{imgs / 'b.jpg'}\n")
    tagger = _wd(tmp_path / "m")
    closed = []
    tagger.close = lambda: closed.append(1)
    monkeypatch.setattr(T, "load", lambda tid, d, log=print: tagger)
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"list_file": str(lst), "trigger": "ohwx", "caption_ext": ".caption",
                               "tags": {"min_score": "0.6", "max_tags": 3}}))
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"backend": "tagger", "tagger": "wd-eva02", "models_dir": str(tmp_path / "m"),
                               "device": "cpu"}))
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdin", io.StringIO(f"RUN {job}\nRUN {job}\n"))
    monkeypatch.setattr(sys, "stdout", out)
    assert B._serve(str(cfg)) == 0
    lines = out.getvalue().splitlines()
    assert lines[-3:] == ["PROGRESS: 2 2", "OK: b.jpg", "DONE"]
    assert lines.count("DONE") == 1 and "READY" in lines and "PROGRESS: 1 2" in lines   # one job, then it quits
    assert closed == [1]
    assert (imgs / "a.caption").read_text(encoding="utf-8") == "ohwx, 1girl, long hair, ^_^"


def test_worker_reports_a_picture_with_no_tags(tmp_path):
    IMG.save(tmp_path / "a.png")
    (tmp_path / "l.txt").write_text(str(tmp_path / "a.png"))
    out = io.StringIO()
    st = B.CaptionState("tagger", "cpu", ("wd-eva02", _wd(tmp_path / "m")), False)
    old, sys.stdout = sys.stdout, out
    try:
        B._run_job(st, {"list_file": str(tmp_path / "l.txt"), "tags": {"min_score": "0.99"}})
    finally:
        sys.stdout = old
    assert "FAIL: a.png (no tag scored at least the min score)" in out.getvalue()
    assert not (tmp_path / "a.txt").exists()


class FakeProc:
    def __init__(self):
        self.stdout, self.stdin, self.alive = iter(["READY\n"]), io.StringIO(), True

    def poll(self):
        return None if self.alive else 0

    def wait(self, timeout=None):
        self.alive = False


@pytest.fixture
def jobdir(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "JOB_DIR", str(tmp_path / "jobs"))
    started = []

    def popen(cmd, prefs, stdin=False):
        started.append(cmd)
        return FakeProc()
    monkeypatch.setattr(C, "_popen", popen)
    yield started
    C._st.update(state="idle", job="", proc=None, key=None, ready=None, model="")


def test_caption_job_overwrite_skip_and_config(tmp_path, jobdir):
    models, imgs = tmp_path / "models", tmp_path / "set"
    imgs.mkdir()
    for n in ("a.png", "b.png", "c.png"):
        IMG.save(imgs / n)
    (imgs / "b.txt").write_text("kept")
    with pytest.raises(ValueError, match="isn't downloaded yet: press Download"):
        C.ai_captions({}, "", str(imgs), T.WD_MODEL, "", "", 120, "ohwx", False, False, models_dir=str(models))
    _wd_folder(models)
    with pytest.raises(ValueError, match="Min score"):
        C.ai_captions({}, "", str(imgs), T.WD_MODEL, "", "", 120, "", False, False, tags={"min_score": "5"},
                      models_dir=str(models))
    r = C.ai_captions({}, "", str(imgs), T.WD_MODEL, "<CAPTION>", "", 120, "ohwx", False, False,
                      tags={"min_score": "0.6", "style": "underscores"}, models_dir=str(models), ext=".txt")
    assert r == {"files": 2}                                          # b already has a caption: skipped
    jd = tmp_path / "jobs"
    cfg = json.loads((jd / "worker_config.json").read_text())
    assert cfg == {"backend": "tagger", "tagger": "wd-eva02", "models_dir": str(models), "include_video": False}
    job = json.loads((jd / "job.json").read_text())
    assert job["tags"]["style"] == "underscores" and job["tags"]["min_score"] == 0.6 and job["trigger"] == "ohwx"
    assert "florence_task" not in job and "instruction_file" not in job
    assert (jd / "images.txt").read_text().split() == [str(imgs / "a.png"), str(imgs / "c.png")]
    # the job through the worker: a and c get tag captions, b's stays
    st = B.CaptionState("tagger", "cpu", ("wd-eva02", _wd(tmp_path / "m2")), False)
    old, sys.stdout = sys.stdout, io.StringIO()
    try:
        B._run_job(st, job)
    finally:
        sys.stdout = old
    assert (imgs / "a.txt").read_text() == "ohwx, 1girl, long_hair, ^_^, smile"
    assert (imgs / "b.txt").read_text() == "kept"
    C._st.update(state="idle", job="")
    with pytest.raises(ValueError, match="JTP-3 Hydra tagger isn't downloaded yet"):
        C.ai_captions({}, "", str(imgs), T.JTP3_MODEL, "", "", 120, "", True, False, models_dir=str(models))
    _jtp3_folder(models)
    C._st.update(state="idle", job="")
    assert C.ai_captions({}, "", str(imgs), T.JTP3_MODEL, "", "", 120, "", True, False,
                         models_dir=str(models)) == {"files": 3}      # overwrite: every picture
    assert json.loads((jd / "job.json").read_text())["tags"]["min_score"] == 0.30
    assert len(jobdir) == 2                                           # a fresh worker for each tagger job


def test_florence_and_qwen_unchanged(tmp_path):
    assert C._worker_config(C.FLORENCE_DEFAULT_MODEL, "", False) == {
        "backend": "florence", "include_video": False, "florence_model": C.FLORENCE_DEFAULT_MODEL,
        "florence_revision": C.FLORENCE_REVISIONS[C.FLORENCE_DEFAULT_MODEL],
        "florence_code_revision": C.FLORENCE_CODE_REVISIONS[C.FLORENCE_DEFAULT_MODEL]}
    assert C._worker_config(C.QWEN_MODEL, "/te.safetensors", True) == {
        "backend": "qwen", "include_video": True, "text_encoder": "/te.safetensors"}
    o = C.options({}, "/te.safetensors", models_dir=str(tmp_path))
    assert o["models"] == C.FLORENCE_MODELS + [C.QWEN_MODEL, T.WD_MODEL, T.JTP3_MODEL]
    assert o["default_model"] == C.QWEN_MODEL and o["tag_max"] == 255
    assert C.options({}, "", models_dir=str(tmp_path))["models"] == C.FLORENCE_MODELS + [T.WD_MODEL, T.JTP3_MODEL]
    tg = o["taggers"]
    assert tg[T.WD_MODEL]["id"] == "wd-eva02" and tg[T.WD_MODEL]["min_score"] == 0.60 and tg[T.WD_MODEL]["path"] == ""
    assert tg[T.JTP3_MODEL]["id"] == "jtp3" and tg[T.JTP3_MODEL]["min_score"] == 0.30 and tg[T.JTP3_MODEL]["gb"] == 1.0
    _wd_folder(tmp_path)          # the folder Trim tags downloads into is the one captions use
    assert C.options({}, "", models_dir=str(tmp_path))["taggers"][T.WD_MODEL]["path"] == os.path.join(str(tmp_path), W.FOLDER)


def test_server_passes_the_tag_settings():
    from fizgig.web import server
    src = open(server.__file__, encoding="utf-8").read()
    assert 'ext=captions.caption_ext(b.get("ext")), tags=b.get("tags")))' in src


JS = r"""
const {JSDOM} = require("jsdom"); const fs = require("fs");
const html = fs.readFileSync(process.env.STATIC + "/index.html", "utf8");
const card = html.match(/<div class="card-title">Settings<\/div>\s*<div class="card-content">\s*<div class="field-hint" id="capFolder">[\s\S]*?Overwrite existing captions[\s\S]*?<\/label>/)[0];
const dom = new JSDOM("<body><div>" + card + "</div></div><input id='capSongWords'><input type='checkbox' id='capSkipCjk'>" +
  "<select id='capPerPage'><option>12</option></select></body>");
global.window = dom.window; global.document = dom.window.document;
const src = fs.readFileSync(process.env.STATIC + "/app.js", "utf8");
const fn = (name) => { const i = src.indexOf("function " + name + "("); const a = src.lastIndexOf("\n", i) + 1;
  return src.slice(a, src.indexOf("\n}\n", i) + 3); };
const line = (re) => src.match(re)[0];
const code = line(/const \$ = \(id\)[^\n]*\n/) + line(/const el = \(tag[\s\S]*?\n};\n/) +
  line(/const qwenOn = [^\n]*\n/) + line(/const qwenTask = [^\n]*\n/) + line(/const capTagger = [^\n]*\n/) +
  line(/const CAP_MODEL_HINT = [\s\S]*?\n};\n/) + line(/const capExt = [\s\S]*?\n};\n/) +
  fn("capShowModel") + fn("capTagState") + fn("capTagRefresh") + fn("capTagBody") + fn("capJobBody") + fn("capSettings") +
  fn("fillTasks");
const posted = [];
const WD = "WD EVA02 Large v3 tagger (anime tags)", JTP = "JTP-3 Hydra tagger (furry / e621 tags)";
const taggers = {[WD]: {id: "wd-eva02", path: "/m/wd", gb: 1.2, label: "WD EVA02-Large tagger v3", min_score: 0.6, fetch: {}},
                 [JTP]: {id: "jtp3", path: "", gb: 1.0, label: "JTP-3 Hydra tagger", min_score: 0.3, fetch: {}}};
const S = {family: "krea2", imageFolder: "/set", values: {},
  cap: {tagMin: {}, tagChar: {}, pictures: true, modelHint: "Qwen3-VL hint",
        opts: {qwen_available: false, qwen_model: "Qwen3-VL 4B (Krea 2 text encoder)", florence_tasks: ["<CAPTION>", "<DETAILED_CAPTION>"],
               qwen_tasks: [], taggers}}};
const api = async (u, b) => { posted.push([u, b]); return u.startsWith("/api/captions/options") ? {taggers} : {}; };
eval(code + "; global.capShowModel = capShowModel; global.capJobBody = capJobBody; global.capSettings = capSettings;");
const $ = (id) => document.getElementById(id);
for (const m of ["MiaoshouAI/Florence-2-base-PromptGen", WD, JTP]) $("capModel").append(new window.Option(m, m));
const shown = (id) => $(id).closest(".field-row") ? $(id).closest(".field-row").style.display !== "none" : null;
const look = () => ({task: shown("capTask"), tokens: shown("capTokens"), box: $("capTagBox").style.display !== "none",
  hint: $("capModelHint").textContent, min: $("capTagMin").value, state: $("capTagState").textContent,
  body: capJobBody(), saved: capSettings()});
const out = {};
$("capModel").value = "MiaoshouAI/Florence-2-base-PromptGen"; fillTasks(); capShowModel(); out.florence = look();
$("capModel").value = WD; capShowModel(); out.wd = look();
$("capTagMin").value = "0.5"; S.cap.tagMin["wd-eva02"] = "0.5";
$("capTagStyle").value = "underscores"; $("capTagRatings").checked = true; $("capTagChar").value = "0.3";
out.wdSet = look();
$("capModel").value = JTP; capShowModel(); out.jtp = look();
$("capTagState").querySelector("button").click();
$("capModel").value = WD; capShowModel(); out.back = look();
setTimeout(() => { out.posted = posted; console.log(JSON.stringify(out)); }, 20);
"""


def test_page_dropdown_wiring():
    from checks.test_val_picker import JSDOM_DIRS
    if not shutil.which("node"):
        pytest.skip("node not installed")
    d = next((p for p in JSDOM_DIRS if os.path.isdir(os.path.join(p, "jsdom"))), None)
    if d is None:
        pytest.skip("jsdom not installed")
    out = json.loads(subprocess.check_output(["node", "-e", JS], text=True,
                                             env={**os.environ, "NODE_PATH": d, "STATIC": os.path.abspath(STATIC)}))
    f = out["florence"]                                        # Florence-2: as before, no tag settings sent
    assert f["task"] and f["tokens"] and not f["box"] and f["hint"] == "Qwen3-VL hint" and "tags" not in f["body"]
    assert f["body"]["task"] == "<DETAILED_CAPTION>" and f["body"]["max_tokens"] == 120
    w = out["wd"]
    assert not w["task"] and not w["tokens"] and w["box"] and w["min"] == "0.60"
    assert "Danbooru" in w["hint"] and "The WD EVA02-Large tagger v3 is downloaded" in w["state"]
    assert w["body"]["tags"] == {"min_score": "0.60", "max_tags": "255", "style": "spaces", "ratings": False,
                                 "char_score": ""}
    assert out["wdSet"]["body"]["tags"] == {"min_score": "0.5", "max_tags": "255", "style": "underscores",
                                            "ratings": True, "char_score": "0.3"}
    assert out["wdSet"]["saved"]["tagMin"] == {"wd-eva02": "0.5"} and out["wdSet"]["saved"]["tagStyle"] == "underscores"
    j = out["jtp"]
    assert j["min"] == "0.30" and "e621" in j["hint"] and "furry" in j["hint"]
    assert "Download the JTP-3 Hydra tagger (1 GB)" in j["state"]
    assert ["/api/prep/tagger_fetch", {"tagger": "jtp3"}] in out["posted"]
    assert out["back"]["min"] == "0.5"                         # each tagger keeps its own min score
