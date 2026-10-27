"""Adaptive LR's Min LR / Max LR: the usual values with Other... for one of your own, exactly as Blocks Swap (the
same "combo" row, the same page code): picking Other shows a box, what you type is the value; a value of your own
comes back from a preset, Last Train or a reload (kept in the dropdown, as Blocks Swap keeps one) and goes to the
trainer as typed; any positive number is accepted (1e-5 style too), anything else - or a Min not below Max - is said
under the box and stops Start. The page side runs the page's own renderRow in jsdom (skipped without node / jsdom)."""
import json, os, shutil, subprocess, sys

import pytest

from checks.test_val_picker import JSDOM_DIRS, STATIC
from fizgig.families import checks, form, launch, registry

KEYS = ("ADAPTIVE_LR_MIN", "ADAPTIVE_LR_MAX")


def _rows(desc):
    return {r["key"]: r for s in form.form(desc)["sections"] for r in s["rows"] if r}


@pytest.mark.parametrize("key", ["moss_image", "krea2", "anima", "klein", "zimage", "flux3", "sdxl"])
def test_rows_are_combos_like_blocks_swap(key):
    desc = registry.get(key)
    rows = _rows(desc)
    if not desc.adaptive_lr:                       # SDXL: no Adaptive LR, so no Min / Max boxes at all
        assert "ADAPTIVE_LR_MIN" not in rows
        return
    assert rows["BLOCKS_SWAP"]["kind"] == "combo"
    assert rows["ADAPTIVE_LR_MIN"]["kind"] == rows["ADAPTIVE_LR_MAX"]["kind"] == "combo"
    assert rows["ADAPTIVE_LR_MIN"]["choices"] == list(form.ADAPTIVE_MIN)        # the same presets as before
    assert rows["ADAPTIVE_LR_MAX"]["choices"] == list(form.ADAPTIVE_MAX)


@pytest.mark.parametrize("raw,want", [("1e-5", 1e-5), ("2.5e-5", 2.5e-5), (" 0.00003 ", 3e-5), ("4E-4", 4e-4),
                                      ("1e-4 (default)", 1e-4), ("abc", None), ("", None), (None, None), ("0", None),
                                      ("-1e-5", None), ("inf", None), ("nan", None)])
def test_lr_value(raw, want):
    got = checks.lr_value(raw)
    assert got == pytest.approx(want) if want is not None else got is None


def test_start_checks():
    on = {"ADAPTIVE_LR": True}
    assert checks.learning_rate_range({**on, "ADAPTIVE_LR_MIN": "2.5e-5", "ADAPTIVE_LR_MAX": "7e-4"}) == []
    assert "positive numbers" in checks.learning_rate_range({**on, "ADAPTIVE_LR_MIN": "x", "ADAPTIVE_LR_MAX": "4e-4"})[0]
    assert "positive numbers" in checks.learning_rate_range({**on, "ADAPTIVE_LR_MIN": "-1e-5",
                                                             "ADAPTIVE_LR_MAX": "4e-4"})[0]
    assert "lower than" in checks.learning_rate_range({**on, "ADAPTIVE_LR_MIN": "5e-4", "ADAPTIVE_LR_MAX": "4e-4"})[0]
    assert "lower than" in checks.learning_rate_range({**on, "ADAPTIVE_LR_MIN": "4e-4", "ADAPTIVE_LR_MAX": "4e-4"})[0]


def test_notes_under_the_boxes():
    moss = registry.get("moss_image")
    st = lambda **v: form.state(moss, {**form.DEFAULTS, "ADAPTIVE_LR": True, **v})
    ok = st(ADAPTIVE_LR_MIN="2.5e-5", ADAPTIVE_LR_MAX="7e-4")
    assert not set(KEYS) & set(ok["hidden"]) and not any(k in ok["notes"] for k in KEYS)
    bad = st(ADAPTIVE_LR_MIN="abc", ADAPTIVE_LR_MAX="0")
    assert "positive number" in bad["notes"]["ADAPTIVE_LR_MIN"] and "positive number" in bad["notes"]["ADAPTIVE_LR_MAX"]
    assert "lower than" in st(ADAPTIVE_LR_MIN="5e-4", ADAPTIVE_LR_MAX="4e-4")["notes"]["ADAPTIVE_LR_MIN"]
    off = form.state(moss, {**form.DEFAULTS, "ADAPTIVE_LR": False, "ADAPTIVE_LR_MIN": "abc"})
    assert set(KEYS) <= set(off["hidden"]) and not any(k in off["notes"] for k in KEYS)


def test_presets_and_last_train_keep_your_value():
    moss = registry.get("moss_image")
    p = {"ADAPTIVE_LR": True, "ADAPTIVE_LR_MIN": "2.5e-5", "ADAPTIVE_LR_MAX": "7e-4"}
    for whole in (False, True):
        got = form.apply_preset(moss, dict(form.DEFAULTS), p, whole=whole)
        assert got["ADAPTIVE_LR_MIN"] == "2.5e-5" and got["ADAPTIVE_LR_MAX"] == "7e-4"
    assert "ADAPTIVE_LR_MIN" not in form.STRICT_CHOICES and "OPTIMIZER_TYPE" in form.STRICT_CHOICES


def test_trainer_gets_your_value(tmp_path):
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
                             "DATASET_CONFIG": str(tmp_path / "dataset.toml"), "OPTIMIZER_TYPE": "adamw8bit",
                             "ADAPTIVE_LR": True, "ADAPTIVE_LR_MIN": "2.5e-5", "ADAPTIVE_LR_MAX": "7e-4"},
                      models=models, python=sys.executable, repo_dir=str(tmp_path), image_folder=str(img),
                      cache_root=str(tmp_path / "cache"), samples_dir=str(tmp_path / "out" / "sample"))
    plan = launch.plan(moss, inp)
    cmd = plan.stages[-1].cmd
    assert cmd[cmd.index("--adaptive_lr_min") + 1] == "2.5e-5" and cmd[cmd.index("--adaptive_lr_max") + 1] == "7e-4"
    assert not any("Adaptive" in p for p in plan.problems)
    src = open(os.path.join(os.path.dirname(form.__file__), "train.py"), encoding="utf-8").read()
    assert 'p.add_argument("--adaptive_lr_min", type=float' in src     # any float reaches AdaptiveLR


HARNESS = r"""
const {JSDOM} = require("jsdom"); const fs = require("fs");
const dom = new JSDOM("<body><div id='host'></div></body>");
global.window = dom.window; global.document = dom.window.document;
const src = fs.readFileSync(process.env.STATIC + "/app.js", "utf8");
const fn = (name) => { const i = src.indexOf("function " + name + "("); const a = src.lastIndexOf("\n", i) + 1;
  return src.slice(a, src.indexOf("\n}\n", i) + 3); };
const code = src.match(/const \$ = \(id\)[^\n]*\n/)[0] + src.match(/const el = \(tag[\s\S]*?\n};\n/)[0] +
  fn("fillCombo") + fn("syncInputs") + fn("renderRow");
const rows = JSON.parse(process.env.ROWS);
const S = {values: JSON.parse(process.env.VALUES), fam: {form: {sections: [{rows}]}}};
const onValue = (r, v) => { S.values[r.key] = v; };
eval(code + "; global.renderRow = renderRow; global.syncInputs = syncInputs;");
const host = document.getElementById("host");
for (const r of rows) host.append(renderRow(r));
const look = (k) => { const s = document.getElementById("f_" + k); const o = s.parentNode.querySelector(".combo-other");
  return {value: s.value, options: [...s.options].map((x) => x.value), other_shown: o.style.display !== "none",
          other: o.value, saved: S.values[k]}; };
const out = {loaded: Object.fromEntries(rows.map((r) => [r.key, look(r.key)]))};
const sel = document.getElementById("f_ADAPTIVE_LR_MAX");
sel.value = "__other"; sel.dispatchEvent(new window.Event("change"));
out.picked_other = look("ADAPTIVE_LR_MAX");
const box = sel.parentNode.querySelector(".combo-other");
box.value = "6.5e-4"; box.dispatchEvent(new window.Event("input"));
out.typed = look("ADAPTIVE_LR_MAX");
sel.value = "2e-4"; sel.dispatchEvent(new window.Event("change"));
out.preset_again = look("ADAPTIVE_LR_MAX");
const lsel = document.getElementById("f_ADAPTIVE_LR_MIN");
lsel.value = "__other"; lsel.dispatchEvent(new window.Event("change"));         // its Other box open...
S.values.ADAPTIVE_LR_MIN = "3.3e-5"; syncInputs(["ADAPTIVE_LR_MIN"]);   // ...then a preset / Last Train with its own value
out.synced = look("ADAPTIVE_LR_MIN");
console.log(JSON.stringify(out));
"""


def test_page_other_box():
    if not shutil.which("node"):
        pytest.skip("node not installed")
    d = next((p for p in JSDOM_DIRS if os.path.isdir(os.path.join(p, "jsdom"))), None)
    if d is None:
        pytest.skip("jsdom not installed (npm i jsdom; FIZGIG_JSDOM=<its node_modules>)")
    rows = _rows(registry.get("moss_image"))
    env = {**os.environ, "NODE_PATH": d, "STATIC": os.path.abspath(STATIC),
           "ROWS": json.dumps([rows[k] for k in (*KEYS, "BLOCKS_SWAP")]),
           "VALUES": json.dumps({"ADAPTIVE_LR_MIN": "2.5e-5", "ADAPTIVE_LR_MAX": "4e-4",
                                 "BLOCKS_SWAP": "Auto (detect from GPU)"})}
    out = json.loads(subprocess.check_output(["node", "-e", HARNESS], text=True, env=env))
    lo, hi, bs = (out["loaded"][k] for k in (*KEYS, "BLOCKS_SWAP"))
    assert lo["value"] == "2.5e-5" and "2.5e-5" in lo["options"] and lo["options"][-1] == "__other"   # yours, kept
    assert hi["value"] == "4e-4" and hi["options"] == list(form.ADAPTIVE_MAX) + ["__other"]          # a preset value
    assert bs["options"][-1] == "__other" and not any(x["other_shown"] for x in (lo, hi, bs))         # as Blocks Swap
    assert out["picked_other"]["other_shown"] and out["picked_other"]["other"] == ""
    assert out["typed"]["saved"] == "6.5e-4"
    assert not out["preset_again"]["other_shown"] and out["preset_again"]["saved"] == "2e-4"
    assert out["synced"]["value"] == "3.3e-5" and out["synced"]["options"][-1] == "__other"
    assert not out["synced"]["other_shown"]                                    # the stale box is put away
