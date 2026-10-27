"""Paired likeness (fizgig/web/sample_pairs.py): prompt hashing, pairing resolution, the launch sidecar, the picker's
listing and the scorer's writes to likeness.json. CPU only, no models (a fake embedder)."""
import json
import os

import numpy as np
from PIL import Image

from fizgig.web import gallery_likeness as gl
from fizgig.web import sample_pairs as sp


def test_hash_is_fnv1a_and_matches_app_js_vectors():
    # vectors computed with app.js promptHash (node) - the two must agree
    assert sp.prompt_hash("") == "811c9dc5"
    assert sp.prompt_hash("a") == "e40c292c"
    assert sp.prompt_hash("  @z4ush, a fox  ") == sp.prompt_hash("@z4ush, a fox")
    assert sp.prompt_hash("fox ü") != sp.prompt_hash("fox u")


def test_prompt_lines_follow_launch_rules():
    assert sp.prompt_lines("a\n\n# note\nb --w 512\n  c ") == ["a", "b", "c"]


def _pairs(*items):
    return json.dumps([{"index": i, "hash": sp.prompt_hash(t), "image": im} for i, t, im in items])


def test_resolve_same_moved_and_stale():
    pairs = _pairs((0, "cat", "1.png"), (1, "dog", "2.png"), (2, "fox", "3.png"))
    ok, stale = sp.resolve(pairs, "dog\n# x\ncat\nfox edited")
    assert {p["image"]: p["index"] for p in ok} == {"1.png": 1, "2.png": 0}     # moved prompts follow their text
    assert [p["image"] for p in stale] == ["3.png"]                              # an edited prompt is not scored
    assert sp.resolve("not json", "a") == ([], [])


def test_resolve_duplicate_prompts_do_not_share_an_index():
    ok, stale = sp.resolve(_pairs((0, "same", "1.png"), (1, "same", "2.png")), "same\nsame")
    assert sorted(p["index"] for p in ok) == [0, 1] and not stale


def test_sidecar_round_trip(tmp_path):
    txt = sp.sidecar_text(_pairs((1, "b", "x.png"), (0, "gone", "y.png")), ["a", "b"], str(tmp_path / "data"))
    (tmp_path / f".pairs_run.json").write_text(txt, encoding="utf-8")
    got = sp.read_sidecar(str(tmp_path), "run")
    assert got == [{"index": 1, "hash": sp.prompt_hash("b"), "image": "x.png",
                    "path": os.path.join(str(tmp_path / "data"), "x.png")}]
    assert sp.read_sidecar(str(tmp_path), "other") == []


def test_launch_writes_the_sidecar_beside_the_prompts(tmp_path):
    from fizgig.families import launch
    plan = type("P", (), {"dirs": [], "files": []})()
    st = {"LORA_NAME": "run", "LORA_OUTPUT_DIR": str(tmp_path), "image_folder": str(tmp_path / "data"),
          "samples_dir": str(tmp_path / "sample"),
          "samples": {"enabled": True, "every": "1", "prompts": ["a", "b"], "pairs": _pairs((1, "b", "x.png"))}}
    from fizgig.families import registry
    desc = registry.get("moss_image") or next(iter(registry.FAMILIES.values()))
    try:
        launch._preview_flags(desc, st, plan, [])
    except Exception:
        pass                         # model paths missing on a checks box; the files are planned before that
    side = dict(plan.files).get(os.path.join(str(tmp_path / "sample"), ".pairs_run.json"))
    assert side and json.loads(side)["pairs"][0]["image"] == "x.png"


def _img(p, v):
    Image.new("RGB", (8, 8), (v, v, v)).save(p)


def test_scoring_writes_paired_and_skips_unpaired(tmp_path):
    s, data = tmp_path / "sample", tmp_path / "data"
    s.mkdir(), data.mkdir()
    _img(data / "x.png", 10)
    (s / ".pairs_run.json").write_text(sp.sidecar_text(_pairs((0, "a", "x.png")), ["a", "b"], str(data)))
    for name in ("run_e000001_00_1_1.png", "run_e000001_01_1_2.png", "run_e000002_00_1_1.png", "note.png"):
        _img(s / name, 20)
    (s / ".prompts_run.json").write_text(json.dumps({"1": ["a", "b"], "2": ["changed live", "b"]}))
    gl._write(str(s), ["f1.png", "f2.png", "f3.png"], "live", {"run_e000001_00_1_1.png": 0.5})   # face scores
    vec = {str(data / "x.png"): np.array([1.0, 0.0])}
    n = sp.score_pass(str(s), embed_fn=lambda p: vec.get(p, np.array([0.6, 0.8])))
    d = gl.read(str(s))
    assert n == 1 and d["paired"] == {"run_e000001_00_1_1.png": {"score": 0.6, "image": "x.png", "index": 0}}
    assert d["scores"] == {"run_e000001_00_1_1.png": 0.5}                        # the face scores stay
    assert sp.score_pass(str(s), embed_fn=lambda p: 1 / 0) == 0                 # scored once, not again
    gl._write(str(s), [], "", {})                                                # the face scorer rewriting...
    assert "run_e000001_00_1_1.png" in gl.read(str(s))["paired"]                 # ...keeps the paired section
    os.remove(s / "run_e000001_00_1_1.png")
    gl.merge_paired(str(s), {}, set(os.listdir(s)))
    assert gl.read(str(s))["paired"] == {}                                       # a deleted preview drops out


def test_picker_lists_the_training_images_with_captions(tmp_path):
    from fizgig.web import server
    _img(tmp_path / "a.png", 1)
    (tmp_path / "a.txt").write_text("a fox", encoding="utf-8")
    got = {}
    h = server.Handler.__new__(server.Handler)
    h.json = lambda data, code=200, extra=(): got.update(data)
    h._captions(str(tmp_path), ".txt", "")
    item = got["items"][0]
    assert item["name"] == "a.png" and item["caption"] == "a fox"


def _plan_files(tmp_path, pairs, existing=False):
    from fizgig.families import launch, registry
    desc = registry.get("moss_image") or next(iter(registry.FAMILIES.values()))
    side = tmp_path / "sample" / ".pairs_run.json"
    if existing:
        side.parent.mkdir(exist_ok=True)
        side.write_text('{"pairs": [{"index": 0}]}')
    plan = type("P", (), {"dirs": [], "files": []})()
    st = {"LORA_NAME": "run", "LORA_OUTPUT_DIR": str(tmp_path), "image_folder": str(tmp_path / "data"),
          "samples_dir": str(tmp_path / "sample"),
          "samples": {"enabled": True, "every": "1", "prompts": ["a", "b"], "pairs": pairs}}
    try:
        launch._preview_flags(desc, st, plan, [])
    except Exception:
        pass
    return dict(plan.files).get(str(side))


def test_off_by_default_writes_no_pairs_file_and_hands_launch_no_pairs():
    from fizgig.families import form, registry
    desc = registry.get("moss_image") or next(iter(registry.FAMILIES.values()))
    assert form.DEFAULTS["SAMPLE_PAIRED"] is False
    vals = {"SAMPLE_PROMPT": "a\nb", "SAMPLE_PAIRS": _pairs((1, "b", "x.png")), "SAMPLE_PAIRED": False}
    st = form.inputs(desc, vals, models={}, python="python", repo_dir=".", image_folder="", cache_root="",
                     samples_dir="")
    assert st["samples"]["pairs"] == ""                                        # off: launch never sees the pairs
    vals["SAMPLE_PAIRED"] = True
    st = form.inputs(desc, vals, models={}, python="python", repo_dir=".", image_folder="", cache_root="",
                     samples_dir="")
    assert st["samples"]["pairs"] == vals["SAMPLE_PAIRS"]


def test_no_pairs_no_file_and_an_old_file_is_emptied(tmp_path):
    assert _plan_files(tmp_path, "") is None                                   # nothing paired: no file at all
    assert json.loads(_plan_files(tmp_path, "", existing=True)) == {"pairs": []}   # an earlier run's: emptied


def test_no_pairs_file_means_no_scoring(tmp_path):
    s = tmp_path / "sample"
    s.mkdir()
    _img(s / "run_e000001_00_1_1.png", 5)
    assert sp.score_pass(str(s), embed_fn=lambda p: 1 / 0) == 0
    sp.kick(str(s))                                                            # no .pairs_ file: no thread
    assert str(s) not in sp._busy and "paired" not in gl.read(str(s))


def test_edit_and_slider_runs_hide_the_paired_tick():
    from fizgig.families import form, registry
    for desc in registry.FAMILIES.values():
        for k, label in form.kinds(desc):
            hidden = set(form.state(desc, {"FAMILY_KIND": label})["hidden"])
            assert ("SAMPLE_PAIRED" in hidden) == (k in ("edit", "slider")), (desc.key, k)


def test_app_js_keeps_pairs_when_toggled_off():
    src = open(os.path.join(os.path.dirname(__file__), "..", "src", "fizgig", "web", "static", "app.js"),
               encoding="utf-8").read()
    body = src[src.index("function pairRows"):src.index("function pairPicked")]
    off = body.index("if (!tick.checked")
    assert "setPairs([])" not in body and "SAMPLE_PAIRS = \"\"" not in body[off:]   # off hides, never clears
    assert 'tick.checked = !!S.values.SAMPLE_PAIRED' in body                    # unset = off


def test_paired_on_stops_face_scoring_and_keeps_its_scores(tmp_path, monkeypatch):
    s, data = tmp_path / "sample", tmp_path / "data"
    s.mkdir(), data.mkdir()
    for n in ("f1.png", "f2.png", "f3.png"):
        _img(data / n, 1)
    gl._write(str(s), ["f1.png", "f2.png", "f3.png"], "live", {"x.png": 0.7})
    started = []
    monkeypatch.setattr(gl, "start", lambda *a: started.append(a) or "ok")
    gl._runs[str(s)] = {"gen": 5, "dataset": str(data)}                         # a face scorer running
    got = gl.state(str(s), str(data), paired=True)
    assert str(s) not in gl._runs and not started                               # stood down, not restarted
    assert got["paired_mode"] is True and got["scores"] == {"x.png": 0.7}       # nothing lost
    assert gl.read(str(s))["baselines"] == ["f1.png", "f2.png", "f3.png"]
    got = gl.state(str(s), str(data), paired=False)                              # switch off: it resumes
    assert started and got["paired_mode"] is False and got["scores"] == {"x.png": 0.7}


def test_gallery_hides_face_likeness_while_paired():
    src = open(os.path.join(os.path.dirname(__file__), "..", "src", "fizgig", "web", "static", "app.js"),
               encoding="utf-8").read()
    body = src[src.index("async function likenessShow"):src.index("async function likenessPicker")]
    assert "paired=${pairedOn ? 1 : 0}" in body
    assert '$("likBtn").disabled = pairedOn;' in body
    assert "const on = d.baselines.length === 3 && !pairedOn;" in body          # no face badges, no chart
    assert "const ps = pairedOn ? (d.paired || {})[fig.title] : null;" in body  # and paired badges only when on


def test_starting_face_likeness_is_refused_while_paired():
    from fizgig.web import server
    src = open(server.__file__, encoding="utf-8").read()
    at = src.index('if path == "/api/likeness":            # three dataset photos')
    assert 'b.get("paired") and b.get("baselines")' in src[at:at + 400]


def test_mid_run_update_writes_the_sidecar_and_scores_existing_previews(tmp_path, monkeypatch):
    s, data = tmp_path / "sample", tmp_path / "data"
    s.mkdir(), data.mkdir()
    _img(data / "x.png", 10), _img(data / "y.png", 30)
    _img(s / "run_e000003_00_1_1.png", 20)
    _img(s / "run_e000003_01_1_2.png", 20)
    kicked = []
    monkeypatch.setattr(sp, "kick", lambda d: kicked.append(d))
    assert sp.update(str(s), "run", "a\nb", "", str(data)) == 0                 # nothing paired: no file
    assert not (s / ".pairs_run.json").exists() and not kicked
    assert sp.update(str(s), "run", "a\nb", _pairs((0, "a", "x.png")), str(data)) == 1
    assert kicked == [str(s)] and sp.read_sidecar(str(s), "run")[0]["image"] == "x.png"
    emb = lambda p: np.array([1.0, 0.0]) if p.endswith("x.png") else np.array([0.0, 1.0]) if p.endswith("y.png") \
        else np.array([0.6, 0.8])
    assert sp.score_pass(str(s), embed_fn=emb) == 1                             # the existing epoch-3 preview
    sp.update(str(s), "run", "a\nb", _pairs((0, "a", "y.png")), str(data))      # re-picked: rescored
    assert sp.score_pass(str(s), embed_fn=emb) == 1
    assert gl.read(str(s))["paired"]["run_e000003_00_1_1.png"] == {"score": 0.8, "image": "y.png", "index": 0}
    sp.update(str(s), "run", "a\nb", "", str(data))                             # switched off: emptied
    assert sp.read_sidecar(str(s), "run") == []


def test_mid_run_update_refuses_odd_runs_and_folders(tmp_path):
    import pytest
    for d, run in ((str(tmp_path / "nope"), "run"), (str(tmp_path), ""), (str(tmp_path), "../x")):
        with pytest.raises(ValueError):
            sp.update(d, run, "a", _pairs((0, "a", "x.png")), str(tmp_path))


def test_sample_pairs_endpoint_and_score_now_are_wired():
    from fizgig.web import server
    src = open(server.__file__, encoding="utf-8").read()
    assert 'if path == "/api/sample_pairs":' in src and "sample_pairs.update(" in src
    js = open(os.path.join(os.path.dirname(__file__), "..", "src", "fizgig", "web", "static", "app.js"),
              encoding="utf-8").read()
    assert 'api("/api/sample_pairs"' in js and 'text: "Score now"' in js


def _app_js():
    return open(os.path.join(os.path.dirname(__file__), "..", "src", "fizgig", "web", "static", "app.js"),
                encoding="utf-8").read()


def test_picking_an_image_never_edits_the_prompt_box():
    js = _app_js()
    body = js[js.index("function pairPicked"):js.index("function captionMatch")]
    assert ".value =" not in body and "SAMPLE_PROMPT =" not in body and "confirm(" not in body
    assert "with this image's caption?" not in js                               # the old replace offer is gone
    rep = js[js.index("function pairReplacePrompt"):js.index("function pairCaptionBadge")]
    assert 'S.values.SAMPLE_PROMPT = all.join("\\n")' in rep and "promptHash(clean)" in rep
    assert "boxes._sync()" in rep                                               # the numbered boxes redraw
    copy = js[js.index("function pairCaptionBadge"):js.index("async function likenessPicker")]
    assert "pairReplacePrompt(host, i, image, cap)" in copy                     # Copy replaces that row's prompt in the box
    assert 'style: close ? "visibility: hidden" : ""' in copy                   # hidden on a match, cell kept


def test_caption_match_measure():
    import shutil
    import subprocess
    import pytest
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    js = _app_js()
    fn = js[js.index("function captionMatch"):js.index("async function pairCaptions")]
    cases = [("A fox  in Snow", "a fox in snow"), ("tag1, tag2, tag3, tag4", "tag4, tag3, tag2, tag1"),
             ("tag1, tag2, tag3, tag4", "tag1, tag2, tag5, tag6"), ("a red fox lying in snow", "a red fox sitting"),
             ("anything", "")]
    out = subprocess.run([node, "-e", fn + f"\nconsole.log(JSON.stringify({json.dumps(cases)}.map(([a, b]) => captionMatch(a, b))))"],
                         capture_output=True, text=True, check=True).stdout
    same, tags_same, tags_half, prose, empty = json.loads(out)
    assert same == 1 and tags_same == 1 and empty is None
    assert abs(tags_half - 2 / 6) < 1e-9                                        # 2 shared of 6 tags
    assert abs(prose - 3 / 7) < 1e-9                                            # a, red, fox of 7 words


def test_paired_badge_fixed_bands_and_best_per_prompt():
    import shutil
    import subprocess
    import pytest
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    js = _app_js()
    fn = js[js.index("const PAIR_COLD"):js.index("// ---- paired likeness (fizgig/web/sample_pairs.py)")]
    def ev(expr):
        return json.loads(subprocess.run([node, "-e", fn + f"\nconsole.log(JSON.stringify({expr}))"],
                                         capture_output=True, text=True, check=True).stdout)
    def run(paired):
        return ev(f"pairBadgeInfo({json.dumps(paired)})")
    DR, R, O, Y, B, G, P = "#A93226", "#FF5A4E", "#FF9F43", "#FFE04A", "#4DA8FF", "#2ECC71", "#B57BFF"
    scores = [0.10, 0.399, 0.40, 0.49, 0.50, 0.59, 0.60, 0.699, 0.70, 0.79, 0.80, 0.899, 0.90, 1.0]
    got = ev(f"{json.dumps(scores)}.map((s) => pairHeat(s).color)")
    assert got == [DR, DR, R, R, O, O, Y, Y, B, B, G, G, P, P]
    assert ev("[0.1 + 0.2 + 0.3, 0.7 + 0.1].map((s) => pairHeat(s).color)") == [Y, G]   # float noise: 0.6000001 / 0.7999999
    assert len(set(ev("PAIR_BANDS.map((b) => b[1])"))) == 7                              # every band its own colour
    assert ev("pairHeat(0.2).text") == "#FFFFFF" and ev("pairHeat(0.85).text") == "#10202A"
    out = run({"f1": {"score": 0.60, "index": 0}, "f2": {"score": 0.65, "index": 0}, "f3": {"score": 0.55, "index": 0},
               "d1": {"score": 0.41, "index": 1}, "d2": {"score": 0.80, "index": 1}, "d3": {"score": 0.90, "index": 1},
               "bad": {"score": None, "index": 0}})
    assert "bad" not in out
    assert out["d1"]["color"] == R and out["d1"]["rank"] == 3
    assert out["d2"]["color"] == G and out["d3"]["color"] == P                    # 0.90+ purple, not green
    assert out["f1"]["color"] == out["f2"]["color"] == Y and out["f3"]["color"] == O   # fixed: no per-run rescaling
    assert out["d3"]["best"] and out["f2"]["best"] and not out["f1"]["best"] and out["f3"]["rank"] == 3 and out["f1"]["n"] == 3
    solo = run({"s": {"score": 0.55, "index": 0}})
    assert solo["s"]["color"] == O and not solo["s"]["best"]
    assert "pairLegend" not in js and "may be overcooked" in js and "fades from" not in js
    assert "pairBadgeInfo(d.paired" in js and "rank ${bi.rank} of ${bi.n}" in js and "b.style.color = bi.text" in js


def test_resolve_pairs_updates_on_manual_or_copied_prompt_edit():
    import shutil
    import subprocess
    import pytest
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    js = _app_js()
    fn = (js[js.index("function promptHash"):js.index("function setPairs")]
          + "\n" + js[js.index("function captionMatch"):js.index("async function pairCaptions")])
    script = fn + r"""
const S = {values: {SAMPLE_PAIRS: JSON.stringify([{index: 0, hash: promptHash("old prompt"), image: "1.png"}])},
           pairCaps: {"1.png": "a red fox in snow"}};
const r1 = resolvePairs("a red fox in snow\nsecond prompt");
S.values.SAMPLE_PAIRS = JSON.stringify([{index: 2, hash: promptHash("gone"), image: "1.png"}]);
const r2 = resolvePairs("first\na red fox in snow");
console.log(JSON.stringify({r1, r2, match: captionMatch("a red fox in snow", S.pairCaps["1.png"])}));
"""
    out = json.loads(subprocess.run([node, "-e", script], capture_output=True, text=True, check=True).stdout)
    assert out["r1"]["ok"]["0"]["hash"] == sp.prompt_hash("a red fox in snow") and not out["r1"]["stale"]
    assert out["r2"]["ok"]["1"]["hash"] == sp.prompt_hash("a red fox in snow") and not out["r2"]["stale"]
    assert out["match"] == 1


def test_pairs_follow_the_numbered_prompt_boxes():
    """Upstream's numbered prompt list (promptLines(r, wrap)) replaced the prompts' textarea: the paired rows hang off
    it, read the prompts from S.values, and the trainer's line rule has its own name (no second promptLines)."""
    import re
    js = _app_js()
    assert len(re.findall(r"^function promptLines\(", js, re.M)) == 1
    assert "function trainerPrompts(raw)" in js
    box = js[js.index("function promptLines(r, wrap)"):js.index("function blockPicker")]
    assert 'r.key === "SAMPLE_PROMPT"' in box and "pairRows(pairs, S.values[r.key]" in box
    pair = js[js.index("function pairRows"):js.index("function captionMatch")]
    assert '$("f_SAMPLE_PROMPT").value' not in pair and "promptLines(" not in pair
