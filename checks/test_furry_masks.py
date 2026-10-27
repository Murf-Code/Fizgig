"""Make masks for drawn and furry pictures: Who's in your pictures (photo / drawn / a mix), the character finder
(SkyTNT anime-seg, dataset/charmask.py) and SAM 3.1's "drawn or furry" tick. No models here: stand-ins answer for
the person finder, the character finder and SAM. Real people must get exactly the masks they got before: the photo
path is compared, file for file and line for line, with the code as it was (e023f5b5's automask.py / sammask.py)."""
import json
import os
import shutil
import subprocess
import types

import numpy as np
import pytest
from PIL import Image

from fizgig.dataset import automask, charmask, sammask

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "..", "src", "fizgig", "web", "static")
ROOT = os.path.join(HERE, "..")
BEFORE = "e023f5b5"


def _pic(path, w=96, h=80, seed=0):
    Image.fromarray(np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)).save(path)


class FakeSegmenter:
    """automask.Segmenter: a person in the left half (face skin at the top, clothes below), per-picture seeded noise
    so masks differ and are soft."""

    def __init__(self, path=None, face=True):
        self.face = face

    def classes(self, img):
        w, h = img.size
        seed = int(np.asarray(img)[0, 0, 0])
        rng = np.random.default_rng(seed)
        c = np.zeros((h, w, 6), np.float32)
        c[..., automask.BACKGROUND] = 1.0
        x2 = w // 2
        c[:, :x2, automask.BACKGROUND] = 0.0
        c[:, :x2, automask.CLOTHES] = 1.0
        if self.face:
            c[: h // 4, :x2, automask.CLOTHES] = 0.0
            c[: h // 4, :x2, automask.FACE] = 0.7
            c[: h // 4, :x2, automask.HAIR] = 0.3
        c[h // 4:h // 2, :x2, automask.CLOTHES] = 0.5
        c[h // 4:h // 2, :x2, automask.BODY] = 0.5
        noise = rng.random((h, w)).astype(np.float32) * 0.2
        c[..., automask.OTHER] += noise
        c /= c.sum(-1, keepdims=True)
        return c


class FakeChars:
    """charmask.CharacterFinder: the character fills the given columns."""

    def __init__(self, x1=0.0, x2=0.5):
        self.x1, self.x2, self.calls = x1, x2, 0

    def mask(self, img):
        self.calls += 1
        w, h = img.size
        m = np.zeros((h, w), np.float32)
        m[:, int(w * self.x1):int(w * self.x2)] = 1.0
        return m


class FakeHeads:
    def __init__(self, boxes):
        self.boxes = boxes

    def detect_from_pil(self, img):
        return [types.SimpleNamespace(bbox=b) for b in self.boxes]


def _old(module):
    src = subprocess.check_output(["git", "-C", ROOT, "show", f"{BEFORE}:src/fizgig/dataset/{module}.py"], text=True)
    mod = types.ModuleType(module + "_old")
    exec(compile(src, module + "_old.py", "exec"), mod.__dict__)
    return mod


def _halves():
    a = np.zeros((60, 80, 3), np.uint8)
    a[:, 40:] = 255
    return Image.fromarray(a)


def _embed(crop):
    return np.array([0.0, 1.0]) if np.asarray(crop).mean() > 128 else np.array([1.0, 0.0])


def _folder_bytes(d):
    return {f: open(os.path.join(d, f), "rb").read() for f in sorted(os.listdir(d))}


# ---- real people: exactly as before ----
@pytest.mark.parametrize("kind", list(automask.KINDS))
@pytest.mark.parametrize("subject", [False, True])
def test_photo_masks_unchanged(tmp_path, monkeypatch, kind, subject):
    try:
        old = _old("automask")
    except Exception:
        pytest.skip("git history not here")
    out = {}
    for name, mod, kw in (("old", old, {}), ("new", automask, {}), ("photo", automask, {"who": "photo"})):
        d = tmp_path / name
        d.mkdir()
        for i in range(3):
            _pic(d / f"p{i}.png", seed=i + 1)
        _pic(d / "keep.png", seed=9)
        Image.new("L", (96, 80), 255).save(d / "keep-masklabel.png")   # a painted mask: kept
        monkeypatch.setattr(mod, "Segmenter", FakeSegmenter)
        monkeypatch.setattr(mod, "ensure_model", lambda models_dir, log=print: "x")
        sub = det = None
        if subject:
            bases = [np.array([1.0, 0.0])]
            sub = (lambda crop: np.array([1.0, 0.1]) if crop.size[0] < 40 else np.array([0.2, 1.0]), bases)
            det = FakeHeads([(4, 2, 20, 18), (30, 2, 70, 30)])
        log = []
        c = mod.make_folder(str(d), kind, str(tmp_path), log=log.append, subject=sub, detector=det, **kw)
        out[name] = ("".join(log).replace(str(d), "D"), c, _folder_bytes(d))
    assert out["old"] == out["new"] == out["photo"]
    assert out["new"][1]["kept"] == 1


def test_sam_photo_path_unchanged():
    try:
        old = _old("sammask")
    except Exception:
        pytest.skip("git history not here")

    class Seg:
        def __init__(self):
            self.calls = []

        def masks(self, image, prompts, threshold=0.5, refine=2, max_each=None):
            self.calls.append((list(prompts), threshold))
            h, w = image.shape[:2]
            out = []
            for p in prompts:
                if p in ("person", "hair"):
                    for x1 in (0, w // 2):
                        m = np.zeros((h, w), bool)
                        m[:, x1:x1 + w // 2 - 2] = True
                        out.append({"prompt": p, "mask": m})
            return out
    img = _halves()
    sub = (_embed, [np.array([0.0, 1.0])])                    # the face on the right is the subject
    det = FakeHeads([(5, 5, 15, 15), (50, 5, 60, 15)])
    for inc, exc in ((["person"], []), (["hair"], ["glasses"]), (["person", "hair"], [])):
        a, b = Seg(), Seg()
        ra = old.picture_mask(a, img, inc, exc, sub, det)
        rb = sammask.picture_mask(b, img, inc, exc, sub, det)
        assert ra[1] == rb[1] and np.array_equal(ra[0], rb[0]) and a.calls == b.calls


# ---- drawn and furry: the character finder ----
def test_character_finder_fits_the_picture_into_its_square():
    seen = {}

    class Sess:
        def get_inputs(self):
            return [types.SimpleNamespace(name="img")]

        def run(self, _out, feed):
            x = feed["img"]
            seen["shape"], seen["x"] = x.shape, x
            y = np.zeros((1, 1, charmask.SIDE, charmask.SIDE), np.float32)
            y[..., :, : charmask.SIDE // 2] = 1.0                          # the left half of the square
            return [y]
    f = charmask.CharacterFinder("unused", session=Sess())
    img = Image.new("RGB", (200, 100), (255, 0, 0))
    m = f.mask(img)
    assert seen["shape"] == (1, 3, charmask.SIDE, charmask.SIDE) and seen["x"].dtype == np.float32
    x = seen["x"][0]
    assert x[0, charmask.SIDE // 2, 10] == pytest.approx(1.0) and x[1].max() == 0     # red, scaled to 0-1
    assert x[0, 10, 10] == 0 and x[0, -10, 10] == 0                                   # padded above and below
    assert m.shape == (100, 200) and m[:, :90].min() > 0.99 and m[:, 110:].max() < 0.01


def test_auto_tells_photos_from_drawings():
    h, w = 80, 100
    photo = FakeSegmenter().classes(Image.new("RGB", (w, h)))
    char = np.zeros((h, w), np.float32)
    char[:, :50] = 1.0
    assert charmask.looks_like_photo(photo, char)
    assert not charmask.looks_like_photo(FakeSegmenter(face=False).classes(Image.new("RGB", (w, h))), char)  # no face skin
    other = np.zeros((h, w), np.float32)
    other[:, 50:] = 1.0                                     # the character finder sees something the person finder doesn't
    assert not charmask.looks_like_photo(photo, other)


def test_character_kinds():
    char = np.zeros((40, 40), np.float32)
    char[:, 10:30] = 1.0
    heads = charmask.head_region((40, 40), [(12, 0, 28, 8)], grow=0.25)
    assert heads[0:10, 8:32].all() and not heads[11:, :].any()
    assert charmask.mask_from(char, "person").sum() == char.sum()
    assert charmask.mask_from(char, "background").sum() == 40 * 40 - char.sum()
    head = charmask.mask_from(char, "head", heads)
    body = charmask.mask_from(char, "body_clothes", heads)
    assert head.sum() + body.sum() == char.sum() and head[:10, 10:30].all()
    assert charmask.mask_from(char, "not_faces", heads).sum() == 40 * 40 - head.sum()
    for k in ("clothes", "body", "no_clothes"):
        with pytest.raises(ValueError, match="real person"):
            charmask.mask_from(char, k)
    assert set(charmask.DRAWN_KINDS) <= set(automask.KINDS) and set(charmask.KIND_LABELS_DRAWN) == set(charmask.DRAWN_KINDS)


def _folder(tmp_path, n=2):
    d = tmp_path / "data"
    d.mkdir()
    for i in range(n):
        _pic(d / f"c{i}.png", seed=i + 20)
    return d


def test_drawn_masks_come_from_the_character_finder(tmp_path, monkeypatch):
    d = _folder(tmp_path)
    monkeypatch.setattr(automask, "Segmenter", lambda p: pytest.fail("the person finder isn't needed for drawn"))
    monkeypatch.setattr(automask, "ensure_model", lambda *a, **k: pytest.fail("nothing to download for drawn"))
    log = []
    c = automask.make_folder(str(d), "person", str(tmp_path), who="drawn", characters=FakeChars(0.25, 0.75),
                             log=log.append)
    assert c["made"] == 2
    m = np.asarray(Image.open(d / "c0-masklabel.png"))
    assert m[:, 30:70].min() == 255 and m[:, :20].max() == 0
    assert "(character)" not in "".join(log)                       # only a mix says which finder each picture got


def test_drawn_head_uses_the_head_finder(tmp_path):
    d = _folder(tmp_path, 1)
    c = automask.make_folder(str(d), "head", str(tmp_path), who="drawn", characters=FakeChars(0.0, 1.0),
                             heads=FakeHeads([(40, 0, 56, 16)]), log=lambda m: None)
    assert c["made"] == 1
    m = np.asarray(Image.open(d / "c0-masklabel.png")) > 0
    assert m[0:16, 40:56].all() and not m[30:, :].any()


def test_nothing_found_writes_no_mask(tmp_path):
    d = _folder(tmp_path, 1)
    log = []
    c = automask.make_folder(str(d), "person", str(tmp_path), who="drawn", characters=FakeChars(0.0, 0.0),
                             log=log.append)
    assert c["no_person"] == 1 and not (d / "c0-masklabel.png").exists() and "no character found" in "".join(log)


def test_a_mix_picks_per_picture(tmp_path, monkeypatch):
    d = tmp_path / "data"
    d.mkdir()
    _pic(d / "a_photo.png", seed=1)
    _pic(d / "b_drawn.png", seed=2)
    faces = {}

    class Seg(FakeSegmenter):
        def classes(self, img):
            self.face = faces[np.asarray(img)[0, 0, 0]]
            return super().classes(img)
    for n in ("a_photo", "b_drawn"):
        faces[np.asarray(Image.open(d / f"{n}.png"))[0, 0, 0]] = n == "a_photo"
    monkeypatch.setattr(automask, "Segmenter", Seg)
    monkeypatch.setattr(automask, "ensure_model", lambda models_dir, log=print: "x")
    for kind, made in (("person", 2), ("clothes", 1)):
        log = []
        c = automask.make_folder(str(d), kind, str(tmp_path), who="auto", characters=FakeChars(0.0, 0.5),
                                 log=log.append, overwrite=True)
        text = "".join(log)
        assert c["made"] == made and "a_photo.png -> a_photo-masklabel.png (photo)" in text
        if kind == "person":
            assert "b_drawn.png -> b_drawn-masklabel.png (character)" in text
        else:
            assert "b_drawn.png: a drawn or furry character - can't tell clothes from fur" in text
    # the photo in a mix gets the person finder's mask, as with Real people
    e = tmp_path / "photo"
    e.mkdir()
    shutil.copy(d / "a_photo.png", e / "a_photo.png")
    automask.make_folder(str(e), "clothes", str(tmp_path), log=lambda m: None)
    assert (e / "a_photo-masklabel.png").read_bytes() == (d / "a_photo-masklabel.png").read_bytes()


def test_quick_checks_before_the_run(tmp_path, monkeypatch):
    from fizgig import user_data
    from fizgig.web import prep_jobs
    monkeypatch.setattr(user_data, "MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(automask, "available", lambda: True)
    monkeypatch.setattr(charmask, "available", lambda: True)
    assert prep_jobs.quick_check("person", "photo") == ""
    assert "who's in your pictures" in prep_jobs.quick_check("person", "cats")
    assert "clothes from fur" in prep_jobs.quick_check("clothes", "drawn")
    assert "press Download" in prep_jobs.quick_check("person", "drawn")
    assert "press Download" in prep_jobs.quick_check("clothes", "auto")
    os.makedirs(charmask.model_dir(str(tmp_path)))
    open(os.path.join(charmask.model_dir(str(tmp_path)), charmask.FILE), "wb").close()
    assert prep_jobs.quick_check("person", "drawn") == "" and prep_jobs.quick_check("clothes", "auto") == ""
    monkeypatch.setattr(automask, "available", lambda: False)
    assert prep_jobs.quick_check("person", "drawn") == ""                      # no ai-edge-litert needed for drawn
    assert "ai-edge-litert" in prep_jobs.quick_check("person", "auto")
    with pytest.raises(ValueError, match="clothes from fur"):
        prep_jobs.start_masks(str(tmp_path), "clothes", who="drawn")


def test_model_pinned_and_found(tmp_path):
    assert charmask.REPO == "skytnt/anime-seg" and charmask.FILE == "isnetis.onnx" and len(charmask.REVISION) == 40
    assert len(charmask.SHA256) == 64 and charmask.SIZE_MB == 176
    assert charmask.find_model(str(tmp_path)) == ""
    os.makedirs(charmask.model_dir(str(tmp_path)))
    p = os.path.join(charmask.model_dir(str(tmp_path)), charmask.FILE)
    open(p, "wb").close()
    assert charmask.find_model(str(tmp_path)) == p
    assert charmask.info(str(tmp_path))["mb"] == 176 and charmask.WHO_DEFAULT == "photo"


# ---- SAM 3.1: drawn or furry ----
def test_sam_drawn_looser_and_outlines_characters():
    class Seg:
        def __init__(self):
            self.calls = []

        def masks(self, image, prompts, threshold=0.5, refine=2, max_each=None):
            self.calls.append((list(prompts), threshold))
            h, w = image.shape[:2]
            out = []
            for p in prompts:
                for x1 in (0, w // 2):
                    m = np.zeros((h, w), bool)
                    m[:, x1:x1 + w // 2 - 2] = True
                    out.append({"prompt": p, "mask": m})
            return out
    img = _halves()
    sub = (_embed, [np.array([0.0, 1.0])])                    # the face on the right is the subject
    det = FakeHeads([(5, 5, 15, 15), (50, 5, 60, 15)])
    seg = Seg()
    m, note = sammask.picture_mask(seg, img, ["tail"], [], sub, det, drawn=True)
    assert seg.calls == [(["tail"], sammask.DRAWN_THRESHOLD), (sammask.DRAWN_WORDS, sammask.DRAWN_THRESHOLD)]
    assert "kept 1 of 2 people: the best match, 100%" in note and m[:, 45:75].all() and not m[:, :30].any()
    seg = Seg()
    sammask.picture_mask(seg, img, ["furry character"], [], sub, det, drawn=True)
    assert seg.calls[1][0] == ["anthro character", "animal character"]       # what's searched already isn't again
    assert sammask.DRAWN_THRESHOLD == 0.4 and sammask.THRESHOLD == 0.5


# ---- the page ----
def test_page_wiring():
    html = open(os.path.join(STATIC, "index.html"), encoding="utf-8").read()
    js = open(os.path.join(STATIC, "app.js"), encoding="utf-8").read()
    server = open(os.path.join(ROOT, "src", "fizgig", "web", "server.py"), encoding="utf-8").read()
    assert 'data-sam="0">Quick finder</button>' in html and "Who's in your pictures:" in html
    assert 'id="maskWho"' in html and 'id="maskCharState"' in html and 'id="maskSamDrawn"' in html
    assert "My pictures are drawn or furry characters" in html
    assert html.index('id="maskKinds"') < html.index('id="maskSamPreview"')     # one preview for both finders
    assert '"/api/masks/char_fetch"' in server and '"/api/masks/quick_preview"' in server
    assert 'who=str(b.get("who") or "photo")' in server and '"drawn": bool(b.get("drawn"))' in server
    assert '"chars": _char_info()' in server and '"drawn_words": sammask.DRAWN_WORDS' in server
    assert "who: maskWho()" in js and "drawn: $(\"maskSamDrawn\").checked" in js and "maskWho: maskWho()" in js


JS = r"""
const {JSDOM} = require("jsdom"); const fs = require("fs");
const html = fs.readFileSync(process.env.STATIC + "/index.html", "utf8");
const card = html.match(/<div class="section-card" id="maskCard">[\s\S]*?<details class="mask-edit"/)[0];
const dom = new JSDOM("<body>" + card + " id=\"maskEdit\"></details></div></div></body>");
global.window = dom.window; global.document = dom.window.document;
const src = fs.readFileSync(process.env.STATIC + "/app.js", "utf8");
const fn = (name) => { const i = src.indexOf("function " + name + "("); const a = src.lastIndexOf("\n", i) + 1;
  return src.slice(a, src.indexOf("\n}\n", i) + 3); };
const consts = (name) => src.match(new RegExp("const " + name + " = \\{[\\s\\S]*?\\};\\n"))[0];
const code = src.match(/const \$ = \(id\)[^\n]*\n/)[0] + src.match(/const el = \(tag[\s\S]*?\n};\n/)[0] +
  consts("MASK_HINTS") + consts("MASK_HINTS_DRAWN") + consts("MASK_WHO_HINTS") +
  ["maskInfo", "maskWho", "maskSamDrawnWords", "maskSamShow", "maskCharShow", "maskBlocked", "maskSamSave",
   "maskKindsShow", "maskShow"].map(fn).join("");
const R = {kinds: {person: "The person (hair to clothes)", head: "Just the face and hair", clothes: "Just the clothes"},
  available: true, pictures: 3, with_mask: 0, baselines: false, faces: true,
  sam: {path: "", fetch: {}, gb: 1.75, drawn_words: ["anthro character", "furry character", "animal character"]},
  chars: {path: "", fetch: {}, mb: 176},
  who: {photo: "Real people (photos)", drawn: "Drawn or furry characters", auto: "A mix: decide for each picture"},
  drawn_kinds: {person: "The whole character (fur, tail and all)", head: "Just the head"}};
global.S = {imageFolder: "/d", prep: {running: false}, prepSaved: {maskKind: "clothes"}};
const saved = [];
global.api = async (url, body) => { if (body) saved.push(body); return url.startsWith("/api/masks/info") ? R : {}; };
global.meOpen = () => {}; global.maskSamPreview = () => {}; global.prepFindShow = () => {};
eval(code + "; Object.assign(global, {maskInfo, maskShow, maskWho, maskBlocked});");
const $ = (id) => document.getElementById(id);
const look = () => ({who: maskWho(), hint: $("maskWhoHint").textContent, run: $("maskRun").disabled,
  block: maskBlocked(R), dl: $("maskCharState").textContent, dlShown: $("maskCharState").style.display !== "none",
  clothes: document.querySelector('input[value="clothes"]').disabled,
  kind: document.querySelector('input[name="maskKind"]:checked').value,
  personLabel: document.querySelector('.mask-kind-label[data-kind="person"]').textContent});
(async () => {
  const out = {};
  await maskInfo(); out.start = look();
  $("maskWho").querySelector('[data-who="drawn"]').click(); out.drawnNoModel = look();
  R.chars.path = "/m/anime-seg/isnetis.onnx"; maskShow(R); out.drawn = look();
  $("maskWho").querySelector('[data-who="auto"]').click(); out.auto = look();
  out.saved = saved.filter((b) => b.prep).pop().prep;
  $("maskMethod").querySelector('[data-sam="1"]').click();
  $("maskSamDrawn").checked = true; $("maskSamDrawn").dispatchEvent(new window.Event("change"));
  out.samWords = $("maskSamInclude").value; out.whoHidden = $("maskWhoBox").style.display === "none";
  $("maskSamDrawn").checked = false; $("maskSamDrawn").dispatchEvent(new window.Event("change"));
  out.samBack = $("maskSamInclude").value;
  console.log(JSON.stringify(out));
})();
"""


def test_page_who_is_in_your_pictures():
    from checks.test_val_picker import JSDOM_DIRS
    if not shutil.which("node"):
        pytest.skip("node not installed")
    d = next((p for p in JSDOM_DIRS if os.path.isdir(os.path.join(p, "jsdom"))), None)
    if d is None:
        pytest.skip("jsdom not installed")
    out = json.loads(subprocess.check_output(["node", "-e", JS], text=True,
                                             env={**os.environ, "NODE_PATH": d, "STATIC": os.path.abspath(STATIC)}))
    s = out["start"]                                     # the default: real people, as before
    assert s["who"] == "photo" and not s["run"] and not s["dlShown"] and s["kind"] == "clothes" and not s["clothes"]
    assert "Google's person finder" in s["hint"] and s["personLabel"] == "The person (hair to clothes)"
    n = out["drawnNoModel"]
    assert n["who"] == "drawn" and n["run"] and "Download the character finder (176 MB)" in n["dl"]
    assert n["clothes"] and n["kind"] == "person"        # clothes is off for drawn: back to the whole character
    assert n["personLabel"] == "The whole character (fur, tail and all)" and "furry art" in n["hint"]
    d = out["drawn"]
    assert not d["run"] and d["block"] == "" and "is downloaded" in d["dl"]
    a = out["auto"]
    assert a["who"] == "auto" and not a["clothes"] and not a["run"] and "both finders" in a["hint"]
    assert out["saved"]["maskWho"] == "auto"
    assert out["samWords"] == "anthro character, furry character, animal character" and out["whoHidden"]
    assert out["samBack"] == "person"
