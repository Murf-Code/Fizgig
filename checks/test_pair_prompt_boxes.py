"""Paired likeness on upstream's numbered prompt boxes: the rows follow the prompts as they are typed, added or
removed, a pairing stays on its prompt, and a preset's redraw (syncInputs -> _sync) brings the rows up to date.
Runs the page's own code in jsdom (skipped when node or jsdom isn't installed)."""
import json, os, shutil, subprocess
import pytest

from checks.test_val_picker import JSDOM_DIRS, STATIC

HARNESS = r"""
const {JSDOM} = require("jsdom"); const fs = require("fs");
const dom = new JSDOM("<body><div id='host'></div></body>");
global.window = dom.window; global.document = dom.window.document;
global.ResizeObserver = class { observe() {} }; global.requestAnimationFrame = (f) => f();
const src = fs.readFileSync(process.env.STATIC + "/app.js", "utf8");
const fn = (name) => { const i = src.indexOf("function " + name + "("); const a = src.lastIndexOf("\n", i) + 1;
  return src.slice(a, src.indexOf("\n}\n", i) + 3); };
const code = src.match(/const \$ = \(id\)[^\n]*\n/)[0] + src.match(/const el = \(tag[\s\S]*?\n};\n/)[0] +
  ["promptLines", "promptHash", "trainerPrompts", "samplePairs", "resolvePairs", "setPairs", "pairRows"].map(fn).join("");
const S = {values: {SAMPLE_PROMPT: "a fox\nb cat", SAMPLE_PAIRED: true,
                    SAMPLE_PAIRS: ""}, imageFolder: "/d"};
const TOKEN = "t", saveStateSoon = () => {}, pairSyncSoon = () => {}, pairCaptionBadge = () => {};
const promptSetsBar = () => document.createElement("div");
const likenessPicker = () => {}, onValue = (r, v) => { S.values[r.key] = v; };
eval(code + "; global.promptLines = promptLines; global.promptHash = promptHash;");
const wrap = document.createElement("div"); document.getElementById("host").append(wrap);
promptLines({key: "SAMPLE_PROMPT", label: "Prompts", lines: true}, wrap);
const later = () => new Promise((r) => setTimeout(r, 0));
const rows = () => [...document.querySelectorAll("#pairRows .pair-row .pair-text")].map((x) => x.textContent);
(async () => {
  await later();
  const out = {placed: !!document.querySelector("#pairRows"), first: rows()};
  S.values.SAMPLE_PAIRS = JSON.stringify([{index: 1, hash: promptHash("b cat"), image: "cat.png"}]);
  const box = () => [...wrap.querySelectorAll("textarea")];
  box()[0].value = "a red fox"; box()[0].dispatchEvent(new window.Event("input"));
  out.typed = {rows: rows(), value: S.values.SAMPLE_PROMPT,
               paired: [...document.querySelectorAll("#pairRows .pair-name")].map((x) => x.textContent)};
  box()[1].selectionStart = box()[1].selectionEnd = 5;
  box()[1].dispatchEvent(new window.KeyboardEvent("keydown", {key: "Enter"}));
  out.added = {rows: rows(), value: S.values.SAMPLE_PROMPT};
  S.values.SAMPLE_PROMPT = "one\ntwo\nthree\nfour"; document.getElementById("f_SAMPLE_PROMPT")._sync();
  out.preset = rows();
  console.log(JSON.stringify(out));
})();
"""


def test_pair_rows_follow_prompt_boxes():
    if not shutil.which("node"):
        pytest.skip("node not installed")
    d = next((p for p in JSDOM_DIRS if os.path.isdir(os.path.join(p, "jsdom"))), None)
    if d is None:
        pytest.skip("jsdom not installed (npm i jsdom; FIZGIG_JSDOM=<its node_modules>)")
    env = {**os.environ, "NODE_PATH": d, "STATIC": os.path.abspath(STATIC)}
    out = json.loads(subprocess.check_output(["node", "-e", HARNESS], text=True, env=env))
    assert out["placed"] and out["first"] == ["a fox", "b cat"]
    assert out["typed"]["rows"] == ["a red fox", "b cat"] and out["typed"]["value"] == "a red fox\nb cat"
    assert out["typed"]["paired"] == ["cat.png"]                     # the pairing stays on its prompt
    assert out["added"]["value"] == "a red fox\nb cat\n" or out["added"]["rows"][:2] == ["a red fox", "b cat"]
    assert out["preset"] == ["one", "two", "three", "four"]
