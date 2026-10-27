"""Batch Size above 1 (MOSS-Image, Anima, SDXL): the per-image loss switches (the loss watch, per-image LR, the look
warm-up, auto-recaption) now work there - each image of a batch keeps its own loss (checks/test_batch_per_image.py) -
so they stay usable: not greyed, not forced off, no hover note, your ticks shown as they are, and the run and its
record take them as ticked. Prodigy no longer warns under Batch Size (per-image LR works with it above 1; its own
rule still greys the look warm-up). The page side runs the page's own refreshState in jsdom (skipped when node or
jsdom isn't installed)."""
import json, os, shutil, subprocess, sys

import pytest

from checks.test_val_picker import JSDOM_DIRS, STATIC
from fizgig.families import form, launch, registry

PER_IMAGE = ("KREA2_LOSS_WATCH", "KREA2_PER_IMAGE_LR", "KREA2_WARMUP_LOOK", "KREA2_AUTO_RECAPTION")
PRODIGY_LINE = "Per-image LR is off with Prodigy at batch size above 1."


def _state(key="moss_image", opt="adamw", bs="1", **extra):
    return form.state(registry.get(key), {**form.DEFAULTS, "OPTIMIZER_TYPE": opt, "DATASET_BATCH_SIZE": bs,
                                          "KREA2_LOSS_WATCH": True, "KREA2_PER_IMAGE_LR": True, **extra})


@pytest.mark.parametrize("key", ["moss_image", "anima", "sdxl"])
def test_usable_above_one(key):
    if not registry.get(key).loss_watch:
        pytest.skip(f"{key} has no per-image loss switches")
    for bs in ("1", "2", "4"):
        st = _state(key, bs=bs)
        assert not set(PER_IMAGE) & set(st["disabled"])
        assert not any(k in st["notes"] or k in st["tips"] or k in st["force"] or k in st["shown"] for k in PER_IMAGE)
    assert form.BATCH_ONE_SWITCHES == ()


def test_batch_off_values():
    moss = registry.get("moss_image")
    on = {**form.DEFAULTS, **{k: True for k in PER_IMAGE}}
    for bs in ("1", "2"):
        assert form.batch_off(moss, {**on, "DATASET_BATCH_SIZE": bs}) == {**on, "DATASET_BATCH_SIZE": bs}
    src = open(server_py := __import__("fizgig.web.server", fromlist=["x"]).__file__, encoding="utf-8").read()
    start = src[src.index("desc, inputs, plan = _plan(family_key, values"):]
    assert start.index("values = form.batch_off(desc, values)") < start.index("history.start(")   # the run's record


def test_prodigy_alongside_batch():
    for bs in ("1", "2"):
        p = _state(opt="prodigy", bs=bs)
        assert "KREA2_PER_IMAGE_LR" not in p["disabled"]                  # usable on Prodigy at any Batch Size
        assert {"LEARNING_RATE", "ADAPTIVE_LR", "KREA2_WARMUP_LOOK"} <= set(p["disabled"])   # Prodigy's own rule
        assert p["force"]["LEARNING_RATE"] == "1.0"
        assert PRODIGY_LINE not in p["notes"].get("DATASET_BATCH_SIZE", "")
        assert "per-image" not in p["notes"].get("DATASET_BATCH_SIZE", "").lower()
    assert _state(opt="prodigy", bs="2")["notes"]["DATASET_BATCH_SIZE"].startswith("Batch Size 2: switched off - DOP")


@pytest.mark.parametrize("bs,sent", [("1", True), ("2", True)])
def test_run_sends_them_at_any_batch(tmp_path, bs, sent):
    moss = registry.get("moss_image")
    img = tmp_path / "imgs"
    img.mkdir()
    (img / "a.png").write_bytes(b"x")
    (img / "a.txt").write_text("zwxem a photo", encoding="utf-8")
    models = {}
    for mf in moss.model_files:
        if mf.required or mf.pref_key == moss.training_adapter:
            f = tmp_path / f"{mf.pref_key}.safetensors"
            f.write_bytes(b"0")
            models[mf.pref_key] = str(f)
    inp = form.inputs(moss, {**form.DEFAULTS, "LORA_NAME": "t", "LORA_OUTPUT_DIR": str(tmp_path / "out"),
                             "DATASET_CONFIG": str(tmp_path / "dataset.toml"), "OPTIMIZER_TYPE": "prodigy",
                             "LEARNING_RATE": "1.0", "OPTIMIZER_ARGS": "", "DATASET_BATCH_SIZE": bs,
                             "KREA2_LOSS_WATCH": True, "KREA2_PER_IMAGE_LR": True},
                      models=models, python=sys.executable, repo_dir=str(tmp_path), image_folder=str(img),
                      cache_root=str(tmp_path / "cache"), samples_dir=str(tmp_path / "out" / "sample"))
    assert inp["loss_watch"]["per_image_lr"] is sent and inp["loss_watch"]["detect"] is sent
    cmd = launch.plan(moss, inp).stages[-1].cmd
    assert ("--per_image_lr" in cmd) is sent and ("--log_per_image_loss" in cmd) is sent


HARNESS = r"""
const {JSDOM} = require("jsdom"); const fs = require("fs");
const keys = JSON.parse(process.env.KEYS);
const dom = new JSDOM("<body><div class='section-card' id='sec-params'>" + keys.map((k) =>
  `<div class="form-row kind-check" data-key="${k}"><label class="checkbox-label"><input type="checkbox" id="f_${k}">` +
  ` ${k}</label><div class="field-hint row-note" id="note_${k}"></div></div>`).join("") + "</div></body>");
global.window = dom.window; global.document = dom.window.document;
const src = fs.readFileSync(process.env.STATIC + "/app.js", "utf8");
const fn = (name) => { const i = src.indexOf("function " + name + "("); const a = src.lastIndexOf("\n", i) + 1;
  return src.slice(a, src.indexOf("\n}\n", i) + 3); };
const code = src.match(/const \$ = \(id\)[^\n]*\n/)[0] + src.match(/const el = \(tag[\s\S]*?\n};\n/)[0] +
  fn("syncInputs") + fn("refreshState");
const states = JSON.parse(process.env.STATES);
let now = null;
const S = {fam: {form: {sections: [{rows: keys.map((k) => ({key: k, label: k}))}]}}, values: {}};
const api = async () => states[now], refreshStateSoon = () => {}, saveStateSoon = () => {}, turboFollow = () => {};
const liveNote = () => {};
eval(code + "; global.refreshState = refreshState;");
for (const k of keys) if (k.startsWith("KREA2")) { S.values[k] = true; document.getElementById("f_" + k).checked = true; }
const look = () => Object.fromEntries(keys.map((k) => { const b = document.getElementById("f_" + k);
  return [k, {disabled: b.disabled, checked: b.checked, title: b.closest(".form-row").title, saved: S.values[k],
              note: document.getElementById("note_" + k).textContent}]; }));
const reload = () => {                // a fresh page: each switch drawn from the saved values, as renderRow does
  for (const k of keys) { const b = document.getElementById("f_" + k); delete b.dataset.shown; b.disabled = false;
    b.checked = !!S.values[k]; }
};
(async () => {
  const out = {};
  for (const name of JSON.parse(process.env.WALK)) {
    if (name === "reload") { reload(); continue; }
    now = name.split("@")[0]; await refreshState(); out[name] = look();
  }
  console.log(JSON.stringify(out));
})();
"""


def test_page_keeps_them_usable_above_one():
    if not shutil.which("node"):
        pytest.skip("node not installed")
    d = next((p for p in JSDOM_DIRS if os.path.isdir(os.path.join(p, "jsdom"))), None)
    if d is None:
        pytest.skip("jsdom not installed (npm i jsdom; FIZGIG_JSDOM=<its node_modules>)")
    keys = list(PER_IMAGE) + ["DATASET_BATCH_SIZE"]
    # the server's own answers, in the order a user would change them: 1 -> 2 -> Prodigy at 2 -> Prodigy at 1 -> 1
    states = {"adamw1": _state(), "adamw2": _state(bs="2"), "prodigy2": _state(opt="prodigy", bs="2"),
              "prodigy1": _state(opt="prodigy", bs="1"), "back1": _state()}
    walk = ["adamw1", "adamw2", "prodigy2", "prodigy1", "back1", "reload", "adamw2@reloaded", "back1@after"]
    env = {**os.environ, "NODE_PATH": d, "STATIC": os.path.abspath(STATIC), "KEYS": json.dumps(keys),
           "STATES": json.dumps(states, default=list), "WALK": json.dumps(walk)}
    out = json.loads(subprocess.check_output(["node", "-e", HARNESS], text=True, env=env))
    for name in out:                                                            # the saved ticks never change
        assert out[name]["KREA2_PER_IMAGE_LR"]["saved"] is True and out[name]["KREA2_LOSS_WATCH"]["saved"] is True
    for name in ("adamw1", "adamw2", "back1", "adamw2@reloaded", "back1@after"):
        assert not any(out[name][k]["disabled"] or out[name][k]["title"] for k in PER_IMAGE)
        assert out[name]["KREA2_PER_IMAGE_LR"]["checked"] and out[name]["KREA2_LOSS_WATCH"]["checked"]
        assert all(out[name][k]["note"] == "" for k in PER_IMAGE)
    for name in ("adamw2", "prodigy2", "adamw2@reloaded"):
        assert PRODIGY_LINE not in out[name]["DATASET_BATCH_SIZE"]["note"]
        assert out[name]["DATASET_BATCH_SIZE"]["note"].startswith("Batch Size 2: switched off - DOP")
    for name in ("prodigy2", "prodigy1"):
        p = out[name]
        assert not p["KREA2_PER_IMAGE_LR"]["disabled"] and p["KREA2_PER_IMAGE_LR"]["checked"]  # usable on Prodigy
        assert p["KREA2_WARMUP_LOOK"]["disabled"]                                          # Prodigy's own rule
    assert out["prodigy1"]["DATASET_BATCH_SIZE"]["note"] == ""
