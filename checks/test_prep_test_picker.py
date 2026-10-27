"""Image Prep's Test button picks its photo in the Likeness picker's thumbnail grid (the one Image Validation uses), not
the generic file browser: images only (.bmp too), one click closes it and runs /api/prep/test on that picture's full path.
The validation picker keeps its own file list and stays open as before. Runs the page's code in jsdom (skipped when node
or jsdom isn't installed)."""
import json, os, re, shutil, subprocess
import pytest

HERE = os.path.dirname(__file__)
STATIC = os.path.join(HERE, "..", "src", "fizgig", "web", "static")
JSDOM_DIRS = [p for p in (os.environ.get("FIZGIG_JSDOM"), "/tmp/jsd/node_modules",
                          os.path.join(HERE, "node_modules")) if p]

HARNESS = r"""
const {JSDOM} = require("jsdom"); const fs = require("fs");
const html = fs.readFileSync(process.env.STATIC + "/index.html", "utf8");
const modal = html.match(/<div class="modal-backdrop" id="likModal">[\s\S]*?\n  <\/div>\n/)[0];
const dom = new JSDOM("<body>" + modal + "<select id='prepFace'><option value='Furry face' selected>f</option></select>" +
  "<input id='prepFind' value=''><div id='prepTestResult'></div></body>");
global.window = dom.window; global.document = dom.window.document;
const src = fs.readFileSync(process.env.STATIC + "/app.js", "utf8");
const fn = (name) => { const i = src.indexOf("function " + name + "("); const a = src.lastIndexOf("\n", i) + 1;
  return src.slice(a, src.indexOf("\n}\n", i) + 3); };
const code = src.match(/const \$ = \(id\)[^\n]*\n/)[0] + src.match(/const el = \(tag[\s\S]*?\n};\n/)[0] +
  src.match(/const LIK_PICS = [^\n]*\n/)[0] + src.match(/const PREP_TEST_PICS = [^\n]*\n/)[0] +
  fn("likenessPicker") + fn("likenessMarks") + fn("prepTest") + fn("prepTestRun");
const names = ["a.png", "a.txt", "b.JPG", "c.jpeg", "d.webp", "e.bmp", "f.gif", "g.mp4"];
const items = names.map((n) => ({name: n, path: "/d/" + n}));
const S = {lik: {picked: []}, prep: {}, imageFolder: "/d", family: "krea2"};
const TOKEN = "t", capExt = () => ".txt", calls = [], picks = [];
const api = async (url, body) => { calls.push({url, body});
  if (url.startsWith("/api/captions")) return {items};
  return {finder: "yolo", image: "data:,", faces: [{gender: "female", area: 1200, selected: true}]}; };
const pick = (...a) => { picks.push(a); };
const prepOther = () => "", prepYoloWords = () => ({words: [], yolo: []});
eval(code + "; Object.assign(global, {likenessPicker, prepTest});");
const grid = () => [...document.querySelectorAll("#likGrid figure")].map((f) => f.dataset.name);
const open = () => document.getElementById("likModal").classList.contains("active");
const sel = () => [...document.querySelectorAll("#likGrid figure.lik-sel")].map((f) => f.dataset.name);
const tick = () => new Promise((r) => setTimeout(r, 0));
(async () => {
  const out = {};
  prepTest(); await tick(); await tick();
  out.test = {open: open(), grid: grid(), pick: picks.length, title: document.getElementById("likTitle").textContent,
              status: document.getElementById("likPickStatus").textContent,
              list: calls[0].url.startsWith("/api/captions?folder=%2Fd")};
  document.querySelector('#likGrid figure[data-name="e.bmp"]').click(); await tick(); await tick();
  const run = calls.find((c) => c.url === "/api/prep/test");
  out.ran = {open: open(), body: run && run.body, result: document.getElementById("prepTestResult").textContent,
             remembered: S.prep.testName};
  prepTest(); await tick(); await tick();
  out.again = [...document.querySelectorAll("#likGrid figure.on")].map((f) => f.dataset.name);
  document.getElementById("likModal").classList.remove("active");          // the ✕
  S.imageFolder = ""; calls.length = 0;
  prepTest(); await tick();
  out.noFolder = {open: open(), calls: calls.length, msg: document.getElementById("prepTestResult").textContent};
  S.imageFolder = "/d";
  let form = ["a.png"];                                                    // Image Validation > your images, after Test
  await likenessPicker({mode: "multi-toggle", title: "V", picked: form, onToggle: (n) => { form = n; }});
  out.val = {grid: grid(), sel: sel(), title: document.getElementById("likTitle").textContent};
  document.querySelector('#likGrid figure[data-name="d.webp"]').click();
  out.val.after = {open: open(), form, sel: sel(), count: document.getElementById("likCount").textContent};
  console.log(JSON.stringify(out));
})();
"""


@pytest.fixture(scope="module")
def run():
    if not shutil.which("node"):
        pytest.skip("node not installed")
    d = next((p for p in JSDOM_DIRS if os.path.isdir(os.path.join(p, "jsdom"))), None)
    if d is None:
        pytest.skip("jsdom not installed (npm i jsdom; FIZGIG_JSDOM=<its node_modules>)")
    env = {**os.environ, "NODE_PATH": d, "STATIC": os.path.abspath(STATIC)}
    return json.loads(subprocess.check_output(["node", "-e", HARNESS], text=True, env=env))


def test_test_opens_the_thumbnail_picker_not_the_file_browser(run):
    t = run["test"]
    assert t["open"] and t["pick"] == 0 and t["list"]
    assert t["grid"] == ["a.png", "b.JPG", "c.jpeg", "d.webp", "e.bmp"]       # images only: no captions, gif or clips
    assert "Image Prep" in t["title"] and t["status"] == "Click one picture"


def test_one_click_closes_and_tests_that_photo(run):
    r = run["ran"]
    assert not r["open"] and r["body"] == {"path": "/d/e.bmp", "face": "Furry face", "find": ""}
    assert "1 face found (yolo)" in r["result"] and r["remembered"] == "e.bmp"
    assert run["again"] == ["e.bmp"]                                          # reopening marks the last tested photo


def test_no_training_folder(run):
    n = run["noFolder"]
    assert not n["open"] and n["calls"] == 0 and n["msg"] == "Choose the training folder on the Start tab first."


def test_validation_picker_unchanged(run):
    v = run["val"]
    assert v["grid"] == ["a.png", "b.JPG", "c.jpeg", "d.webp"]               # its own list: no .bmp, as before
    assert v["sel"] == ["a.png"] and v["title"] == "V"
    assert v["after"] == {"open": True, "form": ["a.png", "d.webp"], "sel": ["a.png", "d.webp"], "count": "2 selected"}


def test_prep_test_no_longer_uses_the_file_browser():
    js = open(os.path.join(STATIC, "app.js"), encoding="utf-8").read()
    body = js[js.index("function prepTest()"):js.index("async function prepTestRun(")]
    assert "likenessPicker(" in body and not re.search(r"\bpick\(", body)
