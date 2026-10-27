"""Include sub-folders sits on the Start tab under the training folder, one tick with its explanation on hover,
saved as before (TRAIN_SUBFOLDERS); the line under it names the Fizgig folders it leaves out."""
import json
import os
import shutil
import subprocess
import sys

import pytest
from PIL import Image

from fizgig.families import form
from fizgig.families.registry import get as get_family

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "..", "src", "fizgig", "web", "static")


def _pic(path, caption="a photo", size=(32, 32)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", size, (90, 40, 20)).save(path)
    if caption is not None:
        with open(os.path.splitext(path)[0] + ".txt", "w", encoding="utf-8") as f:
            f.write(caption)


def test_the_line_under_the_tick_names_what_it_leaves_out(tmp_path):
    d = tmp_path / "data"
    _pic(str(d / "a.png"))
    _pic(str(d / "outfits" / "r.png"))
    _pic(str(d / "originals_2" / "x.png"))
    desc = get_family("krea2")
    on, _act = form.subfolders_note(desc, {"TRAIN_SUBFOLDERS": True}, str(d))
    assert on.startswith("Training on 2 folders, 2 pictures. Left out (Fizgig's own): originals_2.")
    off, _act = form.subfolders_note(desc, {}, str(d))
    assert "Left out" not in off and "1 sub-folder with 1 more picture is not trained on" in off


def _rows(desc):
    return [(s, r) for s in form.form(desc)["sections"] for r in s["rows"] if r and r["key"] == "TRAIN_SUBFOLDERS"]


def test_one_tick_placed_on_the_start_tab_with_its_hint_on_hover():
    from fizgig.families.registry import training_families
    fams = training_families()
    assert fams
    for desc in fams:
        rows = _rows(desc)
        assert len(rows) == 1                                   # only one control
        _sec, r = rows[0]
        assert r["place"] == "start_folder" and r["label"] == "Include sub-folders" and r["kind"] == "check"
        assert r["hint"] == "" and r["tip"] == form.SUBFOLDERS_HINT
        assert "Fizgig's own folders (originals, excluded_by_look, removed) are never trained on" in r["tip"]
    assert form.DEFAULTS["TRAIN_SUBFOLDERS"] is False           # the saved key and value are unchanged
    inp = form.inputs(get_family("krea2"), {**form.DEFAULTS, "TRAIN_SUBFOLDERS": True}, models={}, python=sys.executable,
                      repo_dir=".", image_folder="", cache_root="", samples_dir="")
    assert inp["subfolders"] is True


JS = r"""
const {JSDOM} = require("jsdom"); const fs = require("fs");
const html = fs.readFileSync(process.env.STATIC + "/index.html", "utf8");
const card = html.match(/<div class="section-card accent-card">\s*<div class="card-title">Training folder[\s\S]*?id="folderOptions"><\/div>\s*<\/div>\s*<\/div>/)[0];
const dom = new JSDOM("<body><div id='tab-start'>" + card + "</div><div id='trainingForm'></div></body>");
global.window = dom.window; global.document = dom.window.document;
const src = fs.readFileSync(process.env.STATIC + "/app.js", "utf8");
const fn = (name) => { const i = src.indexOf("function " + name + "("); const a = src.lastIndexOf("\n", i) + 1;
  return src.slice(a, src.indexOf("\n}\n", i) + 3); };
const code = src.match(/const \$ = \(id\)[^\n]*\n/)[0] + src.match(/const el = \(tag[\s\S]*?\n};\n/)[0] +
  fn("renderSections") + fn("renderPlaced") + fn("renderRow");
const FORM = JSON.parse(process.env.FORM);
const S = {fam: {form: FORM}, values: {TRAIN_SUBFOLDERS: false}, family: "krea2"};
const onValue = (r, v) => { S.values[r.key] = v; };
eval(code + "; global.renderSections = renderSections; global.renderPlaced = renderPlaced;");
renderSections("training", document.getElementById("trainingForm"));
renderPlaced("start_folder", document.getElementById("folderOptions"));
const ticks = document.querySelectorAll("#f_TRAIN_SUBFOLDERS");
const t = ticks[0]; const lab = t.closest("label");
const row = t.closest(".form-row");
const out = {count: ticks.length, inStart: !!t.closest("#tab-start"), inTraining: !!t.closest("#trainingForm"),
  afterFolder: !!document.getElementById("imageFolder") && t.closest(".section-card").contains(document.getElementById("imageFolder")),
  label: lab.textContent, title: lab.title, mark: !!lab.querySelector(".tip-mark"),
  hints: [...row.querySelectorAll(".field-hint")].map((h) => h.textContent), note: !!row.querySelector("#note_TRAIN_SUBFOLDERS")};
t.checked = true; t.dispatchEvent(new window.Event("change"));
out.saved = S.values.TRAIN_SUBFOLDERS;
console.log(JSON.stringify(out));
"""


def test_page_draws_the_one_tick_under_the_training_folder():
    from checks.test_val_picker import JSDOM_DIRS
    if not shutil.which("node"):
        pytest.skip("node not installed")
    d = next((p for p in JSDOM_DIRS if os.path.isdir(os.path.join(p, "jsdom"))), None)
    if d is None:
        pytest.skip("jsdom not installed")
    fm = form.form(get_family("krea2"))
    sec = next(s for s in fm["sections"] if any(r and r["key"] == "TRAIN_SUBFOLDERS" for r in s["rows"]))
    fm = {"sections": [{**sec, "rows": [r for r in sec["rows"] if r and r["kind"] in ("check", "note")]}]}
    out = json.loads(subprocess.check_output(["node", "-e", JS], text=True, env={
        **os.environ, "NODE_PATH": d, "STATIC": os.path.abspath(STATIC), "FORM": json.dumps(fm)}))
    assert out["count"] == 1 and out["inStart"] and not out["inTraining"] and out["afterFolder"]
    assert out["label"].strip() == "Include sub-folders ⓘ" and out["mark"]
    assert out["title"].startswith("Off: only the pictures directly in the training folder are used.")
    assert all(h == "" for h in out["hints"]) and out["note"]      # no visible description; its status line stays
    assert out["saved"] is True
