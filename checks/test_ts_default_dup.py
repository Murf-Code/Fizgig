"""Timestep focus shows each preset once: a form saved before the labels changed ("Default", "detail") used to come
back from the web UI's memory as an extra "Default" choice beside "Model default" (a saved value the dropdown didn't
offer was added to it). The form's own list, the saved state the page loads (/api/state), and the page's choice
dropdown and redraw (run in jsdom; skipped without node or jsdom) all map the older names to the labels."""
import json
import os
import shutil
import subprocess

import pytest

from fizgig.families import form
from fizgig.families.registry import FAMILIES, get
from fizgig.training import timestep_spread as TS

HERE = os.path.dirname(__file__)
STATIC = os.path.join(HERE, "..", "src", "fizgig", "web", "static")
SPREAD = [d for d in FAMILIES.values() if getattr(d, "timestep_spread", None)]


def _row(desc):
    return next(r for s in form.form(desc)["sections"] for r in s["rows"] if r and r["key"] == "TIMESTEP_SPREAD")


def test_some_families_have_timestep_focus():
    assert len(SPREAD) >= 5


@pytest.mark.parametrize("desc", SPREAD, ids=lambda d: d.key)
def test_form_lists_each_preset_once(desc):
    r = _row(desc)
    assert r["choices"] == [TS.LABELS[k] for k in desc.timestep_spread]
    assert len(set(r["choices"])) == len(r["choices"])
    assert "Default" not in r["choices"] and "Model default" in r["choices"]
    assert r["default"] == "Model default" and r["default"] in r["choices"]
    a = r["aliases"]
    assert a["default"] == "Model default" and set(a.values()) == set(r["choices"])
    for old in ("Default", "default", " DEFAULT ", "Model default", "detail", "Detail (clean end)"):
        if TS.key(old) in desc.timestep_spread:
            assert form.migrate_values(desc, {"TIMESTEP_SPREAD": old})["TIMESTEP_SPREAD"] in r["choices"]
    assert form.migrate_values(desc, {"TIMESTEP_SPREAD": "Default"}) == {"TIMESTEP_SPREAD": "Model default"}
    assert form.migrate_values(desc, {"TIMESTEP_SPREAD": "nonsense"}) == {}
    assert form.migrate_values(desc, {"X": "Default"}) == {"X": "Default"}      # only Timestep focus


def test_sdxl_has_no_balanced():
    sdxl = get("sdxl")
    if sdxl is None or not sdxl.timestep_spread:
        pytest.skip("no SDXL Timestep focus")
    assert form.migrate_values(sdxl, {"TIMESTEP_SPREAD": "balanced"}) == {}


def test_api_state_maps_a_saved_default(tmp_path, monkeypatch):
    server = pytest.importorskip("fizgig.web.server")
    fam = SPREAD[0].key
    state = {"family": fam, "values": {fam: {"TIMESTEP_SPREAD": "Default", "NETWORK_DIM": "16"},
                                       "not_a_family": {"TIMESTEP_SPREAD": "Default"}}}
    f = tmp_path / "web_state.json"
    f.write_text(json.dumps(state))
    monkeypatch.setattr(server, "STATE_FILE", str(f))
    monkeypatch.setattr(server, "load_last_used", lambda: {})
    st = server.load_state()
    for k, v in (st.get("values") or {}).items():          # what the handler does (kept in step with it below)
        form.migrate_values(server.get_family(k), v)
    assert st["values"][fam] == {"TIMESTEP_SPREAD": "Model default", "NETWORK_DIM": "16"}
    assert st["values"]["not_a_family"] == {"TIMESTEP_SPREAD": "Default"}
    src = open(server.__file__, encoding="utf-8").read()
    i = src.index('if path == "/api/state":\n            st = load_state()')
    assert "form.migrate_values(get_family(fam_key), vals)" in src[i:i + 400]


JS = r"""
const {JSDOM} = require("jsdom"); const fs = require("fs");
const dom = new JSDOM("<body><div id='host'></div></body>");
global.window = dom.window; global.document = dom.window.document;
const src = fs.readFileSync(process.env.STATIC + "/app.js", "utf8");
const fn = (name) => { const i = src.indexOf("function " + name + "("); const a = src.lastIndexOf("\n", i) + 1;
  return src.slice(a, src.indexOf("\n}\n", i) + 3); };
// the choice dropdown as the form draws it: the page's own lines, from its choice branch
const i = src.indexOf('if (r.kind === "choice") {'), j = src.indexOf('} else if (r.kind === "text")', i);
const choiceBody = src.slice(src.indexOf("\n", i) + 1, j);
const code = src.match(/const \$ = \(id\)[^\n]*\n/)[0] + src.match(/const el = \(tag[\s\S]*?\n};\n/)[0] +
  fn("choiceAlias") + fn("fillCombo") + fn("syncInputs") +
  "function drawChoice(r) { const v = S.values[r.key]; let input; const set = () => {};\n" + choiceBody +
  "\nreturn input; }\n";
const ROW = JSON.parse(process.env.ROW);
const S = {values: {}, fam: {form: {sections: [{rows: [ROW]}]}}};
eval(code + "; global.drawChoice = drawChoice; global.syncInputs = syncInputs;");
const out = {};
for (const saved of ["Default", "default", "Model default", "detail", "Composition (noisy end)", undefined]) {
  S.values = saved === undefined ? {} : {TIMESTEP_SPREAD: saved};
  const sel = drawChoice(ROW);
  out[String(saved)] = {opts: [...sel.options].map((o) => o.value), value: sel.value, stored: S.values.TIMESTEP_SPREAD};
}
// Last Train / a run's settings put an older name back in: the redraw shows (and keeps) the label
S.values = {TIMESTEP_SPREAD: "Model default"};
const host = document.getElementById("host"); const sel = drawChoice(ROW); sel.id = "f_TIMESTEP_SPREAD"; host.append(sel);
S.values.TIMESTEP_SPREAD = "Default"; syncInputs(["TIMESTEP_SPREAD"]);
out.sync = {opts: [...sel.options].map((o) => o.value), value: sel.value, stored: S.values.TIMESTEP_SPREAD};
console.log(JSON.stringify(out));
"""


@pytest.mark.parametrize("desc", SPREAD, ids=lambda d: d.key)
def test_page_dropdown_with_a_saved_default(desc):
    from checks.test_val_picker import JSDOM_DIRS
    if not shutil.which("node"):
        pytest.skip("node not installed")
    d = next((p for p in JSDOM_DIRS if os.path.isdir(os.path.join(p, "jsdom"))), None)
    if d is None:
        pytest.skip("jsdom not installed")
    r = _row(desc)
    out = json.loads(subprocess.check_output(["node", "-e", JS], text=True, env={
        **os.environ, "NODE_PATH": d, "STATIC": os.path.abspath(STATIC), "ROW": json.dumps(r)}))
    want = {"Default": "Model default", "default": "Model default", "Model default": "Model default",
            "detail": "Detail (clean end)", "Composition (noisy end)": "Composition (noisy end)"}
    for saved, got in out.items():
        assert got["opts"] == r["choices"], (saved, got)          # each preset once, never a bare "Default"
        assert len(set(got["opts"])) == len(got["opts"]) and "Default" not in got["opts"]
        if saved in want:
            assert got["value"] == want[saved] and got["stored"] == want[saved], (saved, got)
    assert out["sync"]["value"] == "Model default" and out["sync"]["stored"] == "Model default"
