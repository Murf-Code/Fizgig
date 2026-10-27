"""Trim tags that don't fit can use JTP-3 Hydra (RedRocket, Apache-2.0), an e621 tagger for furry art, instead of the
WD EVA02 tagger (still the default, unchanged). The same trim rules: only general tags (e621 category 0) the tagger
knows and scores low go; species, character, artist, copyright, meta (rating), lore, unknown tags, the first tag and
the trigger word always stay. e621 names match with underscores or spaces, older e621 names through RedRocket's rename
list, and a tag implied by one the tagger sees keeps that score. The network is a stand-in here (the real one is a
1.0 GB download), but its layout is checked against the real checkpoint's names and shapes."""
import json
import os
import re
import shutil
import subprocess
import time

import numpy as np
import pytest
import torch
from PIL import Image

from fizgig.dataset import crop_captions as C
from fizgig.dataset import jtp3_tagger as J
from fizgig.dataset import taggers, wd_tagger

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "..", "src", "fizgig", "web", "static")
VENDOR = os.path.join(HERE, "..", "src", "fizgig", "dataset", "jtp3_hydra")

TAGS_CSV = """tag,category,implications
solo,0,
fox,5,canine
canine,5,
tail,0,
breasts,0,
big_breasts,0,breasts
hair,0,
long_hair,0,hair
animal_vulva,0,
krystal_(star_fox),4,
star_fox,3,
safe,7,
explicit,7,
male_(lore),8,
3_heads,0,
"""
ALIASES = "animal_vulva animal_pussy\nbruised_eye black_eye_(injury)\n"
LABELS = ["solo", "fox", "canine", "tail", "breasts", "big_breasts", "hair", "long_hair", "animal_vulva",
          "krystal_(star_fox)", "star_fox", "safe", "explicit", "male_(lore)", "3_heads"]


def _folder(tmp_path):
    d = tmp_path / J.FOLDER
    (d / "data").mkdir(parents=True)
    (d / "models").mkdir()
    (d / J.TAGS).write_text(TAGS_CSV)
    (d / J.ALIASES).write_text(ALIASES)
    (d / J.WEIGHTS).write_text("")
    return str(d)


class FakeNet(torch.nn.Module):
    """JTP-3's face: called with the (1, rows, cols, 768) patch grid, gives each label's probability."""

    def __init__(self, probs):
        super().__init__()
        self.embeds = torch.nn.Module()
        self.embeds.proj = torch.nn.Linear(768, 4)
        self.probs, self.seen = probs, []

    def forward(self, x):
        self.seen.append(tuple(x.shape))
        return torch.tensor([[self.probs.get(t, 0.0) for t in LABELS]])


def _tagger(tmp_path, probs):
    return J.Tagger(_folder(tmp_path), net=FakeNet(probs), labels=LABELS)


def test_registry_keeps_wd_as_the_default():
    assert taggers.DEFAULT == "wd-eva02" and taggers.get(None) is wd_tagger and taggers.get("nope") is wd_tagger
    assert taggers.get("jtp3") is J and taggers.pick(" jtp3 ") == "jtp3"
    assert taggers.NAMES == {"wd-eva02": "WD EVA02 Large v3 (general anime)", "jtp3": "JTP-3 Hydra (furry / e621)"}
    assert wd_tagger.KEEP_SCORE == 0.35 and J.KEEP_SCORE == 0.20


def test_info_lists_both(tmp_path):
    i = taggers.info(str(tmp_path))
    assert set(i) == {"wd-eva02", "jtp3"} and i["jtp3"]["path"] == "" and i["jtp3"]["gb"] == 1.0
    assert i["jtp3"]["keep_score"] == 0.20 and i["wd-eva02"]["keep_score"] == 0.35
    assert i["jtp3"]["name"] == "JTP-3 Hydra (furry / e621)" and i["wd-eva02"]["label"] == wd_tagger.LABEL
    _folder(tmp_path)
    assert taggers.info(str(tmp_path))["jtp3"]["path"] == os.path.join(str(tmp_path), J.FOLDER)


def test_download_pins_the_revision(tmp_path, monkeypatch):
    import huggingface_hub
    got = []

    def fake(repo_id, filename, revision, local_dir):
        got.append((repo_id, filename, revision))
        p = os.path.join(local_dir, filename)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").close()
        return p
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)
    monkeypatch.setitem(J._fetch, "state", "")
    J.download(str(tmp_path))
    for _ in range(100):
        if J.fetch_state(str(tmp_path))["state"] != "running":
            break
        time.sleep(0.02)
    assert J.fetch_state(str(tmp_path))["state"] == "done"
    assert [g[1] for g in got] == [J.ALIASES, J.TAGS, J.WEIGHTS]
    assert {g[0] for g in got} == {"RedRocket/Hydra"} and {g[2] for g in got} == {J.REVISION} and len(J.REVISION) == 40
    assert J.find_model(str(tmp_path)) == os.path.join(str(tmp_path), "jtp-3-hydra")


def test_reads_categories_aliases_and_implications(tmp_path):
    d = _folder(tmp_path)
    meta = J.read_tags(os.path.join(d, J.TAGS))
    assert meta["fox"] == (5, ["canine"]) and meta["solo"] == (0, []) and meta["safe"] == (7, [])
    assert J.read_aliases(os.path.join(d, J.ALIASES)) == {"animal pussy": "animal vulva",
                                                          "black eye (injury)": "bruised eye"}
    assert J.read_aliases(os.path.join(d, "missing.txt")) == {}
    s = J.inherit({"big breasts": 0.9, "breasts": 0.1, "long hair": 0.4, "hair": 0.0, "solo": 0.2},
                  {"big breasts": ["breasts"], "long hair": ["hair"], "a": ["b"], "b": ["a"]})
    assert s == {"big breasts": 0.9, "breasts": 0.9, "long hair": 0.4, "hair": 0.4, "solo": 0.2}


def test_confidence_is_the_demo_scale():
    assert J.Tagger.confidence(0.5) == 0.0 and J.Tagger.confidence(0.3) == 0.0
    assert J.Tagger.confidence(1.0) == 1.0 and abs(J.Tagger.confidence(0.6) - 0.2) < 1e-9


def test_trim_with_jtp3_keeps_species_character_artist_and_meta(tmp_path):
    t = _tagger(tmp_path, {"solo": 0.95, "fox": 0.5, "canine": 0.5, "tail": 0.55, "big_breasts": 0.97,
                           "breasts": 0.52, "long_hair": 0.52, "hair": 0.5, "animal_vulva": 0.9,
                           "krystal_(star_fox)": 0.5, "star_fox": 0.5, "safe": 0.5, "male_(lore)": 0.5,
                           "3_heads": 0.51})
    img = Image.new("RGB", (100, 60), (90, 120, 200))
    s = t.scores(img)
    assert t.net.seen == [(1, 3, 6, 768)]
    assert s["solo"] == pytest.approx(0.9) and s["tail"] == pytest.approx(0.1)
    assert s["breasts"] == pytest.approx(0.94)            # implied by big breasts, which it sees
    assert s["animal pussy"] == s["animal vulva"]           # an older e621 name
    assert t.categories["fox"] == 5 and t.categories["krystal (star fox)"] == 4 and t.categories["safe"] == 7
    assert t.categories["animal pussy"] == 0 and t.categories["star fox"] == 3 and t.categories["male (lore)"] == 8
    cap = ("ruaidri, tail, fox, canine, Krystal_\\(star_fox\\), star fox, safe, male_(lore), by_some_artist, "
           "big breasts, breasts, long_hair, hair, animal_pussy, 3 heads, solo, my own tag")
    out, dropped = wd_tagger.trim(cap, s, t.categories, ["ruaidri"], J.KEEP_SCORE)
    assert dropped == ["tail", "long_hair", "hair", "3 heads"]
    assert out == ("ruaidri, fox, canine, Krystal_\\(star_fox\\), star fox, safe, male_(lore), by_some_artist, "
                   "big breasts, breasts, animal_pussy, solo, my own tag")
    _, dropped = wd_tagger.trim("cat, " + cap, s, t.categories, ["ruaidri"], J.KEEP_SCORE)
    assert "tail" in dropped and "ruaidri" not in dropped and dropped[0] == "tail"   # the first tag (cat) stays


def test_crop_captions_use_the_chosen_tagger(tmp_path):
    t = _tagger(tmp_path, {"solo": 0.95, "tail": 0.51})
    pic, crop = tmp_path / "a.png", tmp_path / "FaceCrop_001.png"
    Image.new("RGB", (64, 64)).save(pic)
    (tmp_path / "a.txt").write_text("me, solo, tail, fox")
    cc = C.CropCaptions(keep=["me"], tagger=t, keep_score=J.KEEP_SCORE)
    cc.to_crop(str(pic), str(crop), lambda m: None, Image.new("RGB", (40, 40)))
    assert (tmp_path / "FaceCrop_001.txt").read_text() == "me, solo, fox"


def test_fit_patches_keeps_aspect_and_never_enlarges():
    assert J.fit_patches(512, 512) == (512, 512)
    assert J.fit_patches(100, 60) == (96, 48)
    assert J.fit_patches(10, 10) == (16, 16)
    for w, h in ((1024, 747), (4000, 3000), (3000, 500), (640, 4096)):
        fw, fh = J.fit_patches(w, h)
        assert fw % 16 == 0 and fh % 16 == 0 and (fw // 16) * (fh // 16) <= 1024 and fw <= w and fh <= h
        assert abs(fw / fh - w / h) / (w / h) < 0.1
    assert (fw // 16) * (fh // 16) > 900                       # most of the 1024-patch budget is used


def test_preprocess_cuts_patches_in_order():
    a = np.zeros((32, 48, 3), dtype=np.uint8)
    a[:16, 16:32] = (255, 0, 0)                               # patch row 0, column 1
    a[16:, 32:] = (0, 0, 255)                                 # patch row 1, column 2
    x = J.preprocess(Image.fromarray(a))
    assert x.shape == (1, 2, 3, 768) and x.dtype == torch.float32
    p = x[0].reshape(2, 3, 16, 16, 3)
    assert torch.all(p[0, 1, ..., 0] == 1) and torch.all(p[0, 1, ..., 1] == -1)
    assert torch.all(p[1, 2, ..., 2] == 1) and torch.all(p[0, 0] == -1)
    clear = J.preprocess(Image.new("RGBA", (16, 16), (255, 255, 255, 0)))
    assert torch.all(clear == -1)                             # transparency is black, as JTP-3 was trained
    white = J.preprocess(Image.new("RGBA", (16, 16), (0, 0, 0, 0)), background=(255, 255, 255))
    assert torch.all(white == 1)


def test_network_layout_matches_the_checkpoint():
    """make_net + convert against the real jtp-3-hydra.safetensors' 336 names and shapes (read off the file)."""
    ck = {"attn_pool.ff_in.weight": (12288, 2048), "attn_pool.ff_norm.bias": (2048,),
          "attn_pool.ff_norm.weight": (2048,), "attn_pool.ff_out.weight": (2048, 6144),
          "attn_pool.kv.weight": (4096, 1152), "attn_pool.out_proj.weight": (7504, 2048, 2),
          "attn_pool.q": (32, 7504, 64), "embeds.pos_embed": (1, 16, 16, 1152), "embeds.proj.bias": (1152,),
          "embeds.proj.weight": (1152, 768), "norm.bias": (1152,), "norm.weight": (1152,)}
    block = {"attn.proj.bias": (1152,), "attn.proj.weight": (1152, 1152), "attn.qkv.bias": (3456,),
             "attn.qkv.weight": (3456, 1152), "mlp.fc1.bias": (4304,), "mlp.fc1.weight": (4304, 1152),
             "mlp.fc2.bias": (1152,), "mlp.fc2.weight": (1152, 4304), "norm1.bias": (1152,),
             "norm1.weight": (1152,), "norm2.bias": (1152,), "norm2.weight": (1152,)}
    for i in range(27):
        ck.update({f"blocks.{i}.{k}": v for k, v in block.items()})
    assert len(ck) == 336
    sd = J.convert({k: torch.empty(v, device="meta") for k, v in ck.items()})
    net = J.make_net(7504, "meta")
    want = {k: tuple(v.shape) for k, v in net.state_dict().items()}
    assert {k: tuple(v.shape) for k, v in sd.items()} == want


def test_small_network_runs():
    """The vendored pool and head run on this torch (a narrow stand-in; the real one is 7504 tags wide)."""
    from fizgig.dataset.jtp3_hydra.pool import HydraPool
    from fizgig.dataset.jtp3_hydra.head import SwiGLUHead
    torch.manual_seed(0)
    pool = HydraPool(5, 64, 16, input_dim=32, mid_blocks=0, ff_dim=96)
    head = SwiGLUHead(5, 64, logit=False)
    y = head(pool(torch.randn(1, 12, 32), None))
    assert y.shape == (1, 5) and torch.all((y >= 0) & (y <= 1))


def test_vendored_code_is_credited_and_python_310_safe():
    init = open(os.path.join(VENDOR, "__init__.py"), encoding="utf-8").read()
    assert "RedRocket/Hydra" in init and "Apache License, Version 2.0" in init and J.REVISION in init
    for f in ("cufork.py", "glu.py", "head.py", "pool.py", "siglip2.py"):
        src = open(os.path.join(VENDOR, f), encoding="utf-8").read()
        code = "\n".join(line.split("#")[0] for line in src.splitlines())    # comments may mention it
        assert not re.search(r"\w\[\*", code)                                # x[*idxs] needs Python 3.11
        assert "from typing import Self\n" not in src.replace("    from typing import Self\n", "")
    notices = open(os.path.join(HERE, "..", "docs", "THIRD_PARTY_NOTICES.md"), encoding="utf-8").read()
    assert "RedRocket/Hydra" in notices and "jtp3_hydra" in notices


def test_page_settings_pick_the_tagger(tmp_path, monkeypatch):
    from fizgig import user_data
    from fizgig.web import prep_jobs
    monkeypatch.setattr(user_data, "MODELS_DIR", str(tmp_path))
    c, folder = prep_jobs.caption_settings({"trim": False, "tagger": "jtp3"})
    assert c["tagger"] == "jtp3" and c["keep_score"] == 0.20 and folder == ""          # its own default
    c, _ = prep_jobs.caption_settings({"trim": False, "tagger": "jtp3", "keep_score": "0.1"})
    assert c["keep_score"] == 0.1
    c, _ = prep_jobs.caption_settings({"trim": False, "tagger": "what"})
    assert c["tagger"] == "wd-eva02" and c["keep_score"] == 0.35
    with pytest.raises(ValueError, match="needs the JTP-3 Hydra tagger: press Download"):
        prep_jobs.caption_settings({"trim": True, "tagger": "jtp3"})
    _folder(tmp_path)
    c, folder = prep_jobs.caption_settings({"trim": True, "tagger": "jtp3"})
    assert folder == os.path.join(str(tmp_path), J.FOLDER)
    with pytest.raises(ValueError, match="needs the WD EVA02"):                     # WD is still separate
        prep_jobs.caption_settings({"trim": True})


def test_run_loads_the_chosen_tagger(tmp_path, monkeypatch):
    from fizgig import face_utils as F
    from fizgig import user_data
    from fizgig.web import prep_jobs
    from checks.test_crop_captions import _Det, _pic
    _folder(tmp_path / "models")
    data = tmp_path / "data"
    data.mkdir()
    _pic(data / "a.png", 8)
    (data / "a.txt").write_text("ruaidri, solo, tail")
    monkeypatch.setattr(user_data, "MODELS_DIR", str(tmp_path / "models"))
    asked, closed = [], []

    class T:
        categories = {"solo": 0, "tail": 0}

        def scores(self, img):
            return {"solo": 0.9, "tail": 0.05}

        def close(self):
            closed.append(1)

    monkeypatch.setattr(prep_jobs, "_tagger", lambda folder, name=None: asked.append((folder, name)) or T())
    monkeypatch.setattr(prep_jobs, "faces_available", lambda: True)
    monkeypatch.setattr(prep_jobs, "_face_detector", lambda: _Det([F._face_info(20, 10, 60, 58)]))
    prep_jobs.start(str(data), "Auto Prep (Face Crops)", 4.2, "Largest Face", 20, False, "",
                    {"copy": True, "trim": True, "tagger": "jtp3", "trigger": "ruaidri"})
    for _ in range(200):
        if prep_jobs.status()["state"] != "running":
            break
        time.sleep(0.05)
    assert prep_jobs.status()["state"] == "done", prep_jobs.status()["log"]
    assert asked == [(os.path.join(str(tmp_path / "models"), J.FOLDER), "jtp3")] and closed == [1]
    caps = [open(os.path.join(data, f)).read() for f in os.listdir(data) if f.endswith(".txt")]
    assert "ruaidri, solo" in caps


def test_server_and_page_wiring():
    from fizgig.web import server
    src = open(server.__file__, encoding="utf-8").read()
    assert 'taggers.get(b.get("tagger")).download(user_data.MODELS_DIR)' in src
    assert '"taggers": _taggers_info()' in src and '"tagger": _tagger_info()' in src
    html = open(os.path.join(STATIC, "index.html"), encoding="utf-8").read()
    js = open(os.path.join(STATIC, "app.js"), encoding="utf-8").read()
    assert '<option value="wd-eva02">WD EVA02 Large v3 (general anime)</option>' in html
    assert '<option value="jtp3">JTP-3 Hydra (furry / e621)</option>' in html
    assert html.index('id="prepTagger"') < html.index('id="prepKeepScore"') < html.index('id="prepTaggerState"')
    assert "tagger: p.tagger, keep_score: p.keepScore" in js
    assert '$("prepTagger").value = saved.tagger in TAGGER_KEEP ? saved.tagger : "wd-eva02";' in js
    assert 'api("/api/prep/tagger_fetch", {tagger: id})' in js


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
  fn("prepTrimBlocked") + fn("prepFindBlocked") + "function prepBlocked() { return prepFindBlocked() || prepTrimBlocked(); }\n" +
  fn("prepFindShow") + fn("prepTaggerShow") + fn("prepTaggerInfo") + fn("prepTaggerPicked") +
  src.match(/const prepTaggerName = [^\n]*\n/)[0] + src.match(/const TAGGER_KEEP = [^\n]*\n/)[0] +
  src.match(/const TAGGER_HINTS = [\s\S]*?\n\};\n/)[0];
const posted = [];
const S = {prep: {opts: {other: "Other…", face_tools: true}, running: false,
                   maskOpts: {sam: {path: ""}, tagger: {path: "/m/wd", gb: 1.2, fetch: {}},
                              taggers: {"wd-eva02": {path: "/m/wd", gb: 1.2, fetch: {}}, "jtp3": {path: "", gb: 1.0, fetch: {}}}}}};
const api = async (u, b) => { posted.push([u, b]); return {}; }; const maskInfo = () => {};
eval(code + "; global.prepFindShow = prepFindShow; global.prepBlocked = prepBlocked; global.prepSettings = prepSettings; global.prepTaggerPicked = prepTaggerPicked;");
const $ = (id) => document.getElementById(id);
$("prepFace").append(new window.Option("Largest Face", "Largest Face"));
document.querySelector('input[name="prepMode"][value="Face Crop Only"]').checked = true;
const look = () => ({run: $("prepRun").disabled, note: $("prepTrimNote").textContent, hint: $("prepTaggerHint").textContent,
  state: $("prepTaggerState").textContent, keep: $("prepKeepScore").value, settings: prepSettings()});
const out = {};
$("prepCapTrim").checked = true; prepFindShow(); out.wd = look();
out.options = [...$("prepTagger").options].map((o) => [o.value, o.text]);
$("prepTagger").value = "jtp3"; prepTaggerPicked(); prepFindShow(); out.jtp3 = look();
$("prepTaggerState").querySelector("button").click();
$("prepKeepScore").value = "0.15"; $("prepTagger").value = "wd-eva02"; prepTaggerPicked(); prepFindShow(); out.typed = look();
$("prepTagger").value = "jtp3"; prepTaggerPicked(); S.prep.maskOpts.taggers.jtp3.path = "/m/jtp3"; prepFindShow(); out.ready = look();
$("prepKeepScore").value = "0.20"; $("prepTagger").value = "wd-eva02"; prepTaggerPicked(); prepFindShow(); out.back = look();
delete S.prep.maskOpts.taggers; $("prepTagger").value = "jtp3"; prepFindShow(); out.oldServer = look();
setTimeout(() => { out.posted = posted; console.log(JSON.stringify(out)); }, 20);
"""


def test_page_tagger_dropdown():
    from checks.test_val_picker import JSDOM_DIRS
    if not shutil.which("node"):
        pytest.skip("node not installed")
    d = next((p for p in JSDOM_DIRS if os.path.isdir(os.path.join(p, "jsdom"))), None)
    if d is None:
        pytest.skip("jsdom not installed")
    out = json.loads(subprocess.check_output(["node", "-e", JS], text=True,
                                             env={**os.environ, "NODE_PATH": d, "STATIC": os.path.abspath(STATIC)}))
    assert out["options"] == [["wd-eva02", "WD EVA02 Large v3 (general anime)"], ["jtp3", "JTP-3 Hydra (furry / e621)"]]
    wd = out["wd"]                                            # the default, as before
    assert wd["settings"]["tagger"] == "wd-eva02" and wd["keep"] == "0.35" and not wd["run"] and wd["note"] == ""
    assert "Danbooru" in wd["hint"] and "The WD tagger is downloaded" in wd["state"]
    j = out["jtp3"]
    assert j["keep"] == "0.20" and j["settings"]["tagger"] == "jtp3" and j["run"]
    assert "Trim tags needs the JTP-3 Hydra tagger: press Download" in j["note"]
    assert "Download the JTP-3 Hydra tagger (1 GB)" in j["state"] and "e621" in j["hint"] and "furry" in j["hint"]
    assert ["/api/prep/tagger_fetch", {"tagger": "jtp3"}] in out["posted"]
    assert out["typed"]["keep"] == "0.15" and out["typed"]["settings"]["tagger"] == "wd-eva02"   # a typed score stays
    r = out["ready"]
    assert r["keep"] == "0.15" and not r["run"] and r["note"] == "" and "JTP-3 Hydra tagger is downloaded" in r["state"]
    assert out["back"]["keep"] == "0.35"
    assert out["oldServer"]["note"] == ""          # a server without "taggers": the run itself says what's missing
