"""The Likeness picker's multi-toggle mode (Image Validation > Your images): it stays open, each click puts a picture in
or out, previous picks show selected on reopening - and paired likeness' single pick still closes on its click.
Runs the page's own code in jsdom (skipped when node or jsdom isn't installed)."""
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
const dom = new JSDOM("<body>" + modal + "</body>");
global.window = dom.window; global.document = dom.window.document;
const src = fs.readFileSync(process.env.STATIC + "/app.js", "utf8");
const fn = (name) => { const i = src.indexOf("function " + name + "("); const a = src.lastIndexOf("\n", i) + 1;
  return src.slice(a, src.indexOf("\n}\n", i) + 3); };
const code = src.match(/const \$ = \(id\)[^\n]*\n/)[0] + src.match(/const el = \(tag[\s\S]*?\n};\n/)[0] +
  src.match(/const LIK_PICS = [^\n]*\n/)[0] + fn("trainImagesUrl") + fn("likenessPicker") + fn("likenessMarks");
const items = ["a.png", "b.png", "c.png", "d.jpg"].map((n) => ({name: n, path: "/d/" + n}));
const S = {lik: {picked: []}, imageFolder: "/d", family: "krea2"};
const TOKEN = "t", capExt = () => ".txt", api = async () => ({items});
eval(code + "; global.likenessPicker = likenessPicker;");
const fig = (n) => document.querySelector(`#likGrid figure[data-name="${n}"]`);
const open = () => document.getElementById("likModal").classList.contains("active");
const sel = () => [...document.querySelectorAll("#likGrid figure.lik-sel")].map((f) => f.dataset.name);
(async () => {
  const out = {}; let form = ["b.png"];
  const multi = () => likenessPicker({mode: "multi-toggle", title: "V", picked: form, onToggle: (n) => { form = n; }});
  await multi();
  out.preselected = sel(); out.count0 = document.getElementById("likCount").textContent;
  fig("a.png").click(); fig("c.png").click();
  out.afterAdd = {open: open(), form, sel: sel(), count: document.getElementById("likCount").textContent,
                  checks: document.querySelectorAll("#likGrid .lik-check").length};
  fig("b.png").click();
  out.afterRemove = {open: open(), form, sel: sel()};
  document.getElementById("likModal").classList.remove("active");          // the ✕
  await multi();
  out.reopen = {sel: sel(), count: document.getElementById("likCount").textContent};
  let picked = null;
  await likenessPicker({current: null, onPick: (it) => { picked = it.name; }});
  out.singleCountHidden = document.getElementById("likCount").style.display === "none";
  fig("d.jpg").click();
  out.single = {open: open(), picked, sel: sel()};
  await likenessPicker();                                                   // the Likeness card's pick 3
  for (const n of ["a.png", "b.png", "c.png", "d.jpg"]) fig(n).click();
  out.three = {open: open(), on: [...document.querySelectorAll("#likGrid figure.on")].map((f) => f.dataset.name),
               nums: document.querySelectorAll("#likGrid .lik-num").length, sel: sel(),
               status: document.getElementById("likPickStatus").textContent,
               start: document.getElementById("likStart").disabled};
  console.log(JSON.stringify(out));
})();
"""


def _run():
    if not shutil.which("node"):
        pytest.skip("node not installed")
    d = next((p for p in JSDOM_DIRS if os.path.isdir(os.path.join(p, "jsdom"))), None)
    if d is None:
        pytest.skip("jsdom not installed (npm i jsdom; FIZGIG_JSDOM=<its node_modules>)")
    env = {**os.environ, "NODE_PATH": d, "STATIC": os.path.abspath(STATIC)}
    return json.loads(subprocess.check_output(["node", "-e", HARNESS], text=True, env=env))


@pytest.fixture(scope="module")
def run():
    return _run()


def test_multi_toggle_preselected(run):
    assert run["preselected"] == ["b.png"] and run["count0"] == "1 selected"


def test_multi_toggle_stays_open_and_adds(run):
    a = run["afterAdd"]
    assert a["open"] and a["form"] == ["b.png", "a.png", "c.png"]
    assert sorted(a["sel"]) == ["a.png", "b.png", "c.png"] and a["count"] == "3 selected" and a["checks"] == 3


def test_multi_toggle_removes(run):
    a = run["afterRemove"]
    assert a["open"] and a["form"] == ["a.png", "c.png"] and sorted(a["sel"]) == ["a.png", "c.png"]


def test_reopen_shows_previous_picks(run):
    assert sorted(run["reopen"]["sel"]) == ["a.png", "c.png"] and run["reopen"]["count"] == "2 selected"


def test_paired_likeness_single_pick_still_closes(run):
    assert run["single"] == {"open": False, "picked": "d.jpg", "sel": []} and run["singleCountHidden"]


def test_likeness_pick_three_unchanged(run):
    t = run["three"]
    assert t["open"] and sorted(t["on"]) == ["b.png", "c.png", "d.jpg"] and t["nums"] == 3 and t["sel"] == []
    assert t["status"] == "3/3 selected" and t["start"] is False


EVERY = r"""
const {JSDOM} = require("jsdom"); const fs = require("fs");
const dom = new JSDOM("<body></body>"); global.window = dom.window; global.document = dom.window.document;
const src = fs.readFileSync(process.env.STATIC + "/app.js", "utf8");
const fn = (name) => { const i = src.indexOf("function " + name + "("); const a = src.lastIndexOf("\n", i) + 1;
  return src.slice(a, src.indexOf("\n}\n", i) + 3); };
const code = src.match(/const el = \(tag[\s\S]*?\n};\n/)[0] + fn("valEveryRow");
const S = {values: {}}; const saved = [];
const onValue = (r, v) => { S.values[r.key] = v; saved.push(v); };
eval(code + "; global.valEveryRow = valEveryRow;");
const row = {key: "VALIDATION_EVERY_STEPS", label: "How often to check", default: "0", n_default: "500", hint: "h"};
const look = (w) => { const [e, s] = w.querySelectorAll('input[type="radio"]'); const n = w.querySelector(".cad-num");
  return {epoch: e.checked, steps: s.checked, n: n.value, disabled: n.disabled}; };
const out = {};
const fresh = (v) => { S.values = v === undefined ? {} : {VALIDATION_EVERY_STEPS: v}; delete S._valN; saved.length = 0;
  const w = document.createElement("div"); document.body.append(w); return valEveryRow(row, w); };
let w = fresh("0"); out.old0 = look(w);
w.querySelectorAll('input[type="radio"]')[1].click(); out.toSteps = {...look(w), value: S.values.VALIDATION_EVERY_STEPS};
const n = w.querySelector(".cad-num"); n.value = "-3"; n.dispatchEvent(new window.Event("input"));
out.bad = {value: S.values.VALIDATION_EVERY_STEPS, msg: w.querySelector(".row-note").textContent};
n.value = "250"; n.dispatchEvent(new window.Event("input")); out.typed = S.values.VALIDATION_EVERY_STEPS;
w.querySelectorAll('input[type="radio"]')[0].click(); out.backEpoch = {...look(w), value: S.values.VALIDATION_EVERY_STEPS};
w.querySelectorAll('input[type="radio"]')[1].click(); out.remembers = S.values.VALIDATION_EVERY_STEPS;
w = fresh("200"); out.old200 = look(w);
w = fresh(undefined); out.none = look(w);
console.log(JSON.stringify(out));
"""


def test_how_often_to_check_row():
    if not shutil.which("node"):
        pytest.skip("node not installed")
    d = next((p for p in JSDOM_DIRS if os.path.isdir(os.path.join(p, "jsdom"))), None)
    if d is None:
        pytest.skip("jsdom not installed")
    out = json.loads(subprocess.check_output(["node", "-e", EVERY], text=True,
                                             env={**os.environ, "NODE_PATH": d, "STATIC": os.path.abspath(STATIC)}))
    assert out["old0"] == {"epoch": True, "steps": False, "n": "500", "disabled": True}        # old 0 = Every epoch
    assert out["toSteps"] == {"epoch": False, "steps": True, "n": "500", "disabled": False, "value": "500"}
    assert out["bad"]["value"] == "500" and "above 0" in out["bad"]["msg"]                    # >0 only
    assert out["typed"] == "250"
    assert out["backEpoch"] == {"epoch": True, "steps": False, "n": "250", "disabled": True, "value": "0"}
    assert out["remembers"] == "250"
    assert out["old200"] == {"epoch": False, "steps": True, "n": "200", "disabled": False}   # old N = Every N steps
    assert out["none"]["epoch"] and out["none"]["n"] == "500"
