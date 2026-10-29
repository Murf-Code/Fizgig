"""Paired likeness for style LoRAs: a sample prompt can be tied to one training image (picked on the Samples tab with
the Likeness picker), and every preview of that prompt is scored against its own image - whole-image DINOv2 (the same
pinned model as the look match), on the CPU. Prompts without an image stay unscored.

Pairings live in the settings as SAMPLE_PAIRS, a JSON list of {"index", "hash", "image"}: `index` counts the prompt
lines the trainer reads (blank and # lines skipped), `hash` is FNV-1a of that line (the app computes the same), so an
edited prompt is never scored against the wrong image and a moved one follows its text. At launch the resolved
pairings are written beside the previews as `.pairs_<run>.json` (like the trainer's `.prompts_<run>.json`), so they
survive a resume and the gallery can score without the form. Scores go to likeness.json's "paired" section:
{"<sample file>": {"score": 0.81, "image": "0123.png", "index": 0}}."""

import json
import os
import re
import threading

PICS = (".png", ".jpg", ".jpeg", ".webp")
_SAMPLE = re.compile(r"^(?P<run>.+)_e(?P<epoch>\d{6})_(?P<idx>\d{2})_")
_lock = threading.Lock()
_busy = set()
_model = {}


def prompt_hash(text) -> str:
    """FNV-1a 32-bit over the stripped line's UTF-8 bytes (app.js promptHash gives the same)."""
    h = 0x811C9DC5
    for b in str(text or "").strip().encode("utf-8"):
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return f"{h:08x}"


def prompt_lines(raw):
    """The lines the trainer reads, in order (launch.py's rule: blank and # lines out, inline ' --' flags off)."""
    lines = raw.splitlines() if isinstance(raw, str) else list(raw or [])
    out = []
    for ln in lines:
        ln = str(ln).strip()
        if ln and not ln.startswith("#"):
            ln = ln.split(" --")[0].strip()
            if ln:
                out.append(ln)
    return out


def parse(value):
    try:
        v = json.loads(value) if isinstance(value, str) else value
    except ValueError:
        return []
    return [p for p in (v or []) if isinstance(p, dict) and p.get("image") and p.get("hash")] if isinstance(v, list) else []


def resolve(pairs, prompts):
    """Match saved pairings to the prompts as they are now: same index and hash -> ok; the hash at another index ->
    follows the text there; otherwise stale (not scored). Returns ([{index, hash, image}], [stale pairs])."""
    hashes = [prompt_hash(p) for p in prompt_lines(prompts)]
    ok, stale, used = [], [], set()
    for p in parse(pairs):
        i = p.get("index")
        if isinstance(i, int) and 0 <= i < len(hashes) and hashes[i] == p["hash"] and i not in used:
            j = i
        else:
            j = next((k for k, h in enumerate(hashes) if h == p["hash"] and k not in used), None)
        if j is None:
            stale.append(p)
        else:
            used.add(j)
            ok.append({"index": j, "hash": p["hash"], "image": p["image"]})
    return sorted(ok, key=lambda p: p["index"]), stale


def dataset_rel(name):
    """A picked training image -> its path from the training folder, "/"-separated: a bare name stays a name, an
    Include sub-folders pick keeps its "sub/" ("sub/0001.png"); a drive, a leading slash, "." and ".." are dropped,
    so it never leaves the folder."""
    parts = str(name or "").replace("\\", "/").split("/")
    return "/".join(x for x in parts if x not in ("", ".", "..") and ":" not in x)


def sidecar(samples_dir, run):
    return os.path.join(samples_dir, f".pairs_{run}.json")


def sidecar_text(pairs, prompts, dataset_dir, style_dirs=None):
    """What launch writes: the resolved pairings with the image's full path; style_dirs: the training folders the
    Style score compares each preview with (the Style score tick on), else none."""
    ok, _ = resolve(pairs, prompts)
    for p in ok:
        p["path"] = os.path.join(dataset_dir or "", *dataset_rel(p["image"]).split("/"))
    return json.dumps({"pairs": ok, **({"style": [str(d) for d in style_dirs if d]} if style_dirs else {})})


def read_style_dirs(samples_dir, run):
    try:
        with open(sidecar(samples_dir, run), encoding="utf-8") as f:
            d = json.load(f).get("style") or []
        return [str(x) for x in d if isinstance(x, str)]
    except (OSError, ValueError, AttributeError):
        return []


def read_sidecar(samples_dir, run):
    try:
        with open(sidecar(samples_dir, run), encoding="utf-8") as f:
            d = json.load(f)
        return [p for p in d.get("pairs") or [] if isinstance(p, dict)]
    except (OSError, ValueError, AttributeError):
        return []


def _epoch_prompts(samples_dir, run):
    try:
        with open(os.path.join(samples_dir, f".prompts_{run}.json"), encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def embed(path):
    """Whole-image DINOv2-base CLS (face_utils' pinned model and preprocessing), L2-normalised."""
    import numpy as np
    import torch
    from PIL import Image
    from fizgig import face_utils as fu
    if "m" not in _model:
        from transformers import AutoModel
        _model["m"] = fu._hf_pretrained(AutoModel, fu._DinoV2.model_id, fu._DinoV2.revision).to("cpu", torch.float32).eval()
    x = fu._preprocess(Image.open(path), 256, 224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    with torch.no_grad():
        v = _model["m"](pixel_values=torch.from_numpy(x)).pooler_output[0].float().numpy()
    return v / (np.linalg.norm(v) + 1e-8)


def style_embed(path):
    """The Style score's vector: the mean and spread of DINOv2-base's patch tokens (texture, palette, line and
    medium more than who is in the picture), L2-normalised. Same model and pass as embed()."""
    import numpy as np
    import torch
    from PIL import Image
    from fizgig import face_utils as fu
    if "m" not in _model:
        from transformers import AutoModel
        _model["m"] = fu._hf_pretrained(AutoModel, fu._DinoV2.model_id, fu._DinoV2.revision).to("cpu", torch.float32).eval()
    x = fu._preprocess(Image.open(path), 256, 224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    with torch.no_grad():
        pt = _model["m"](pixel_values=torch.from_numpy(x)).last_hidden_state[0, 1:].float()
    v = torch.cat([pt.mean(0), pt.std(0)]).numpy()
    return v / (np.linalg.norm(v) + 1e-8)


STYLE_MAX_IMAGES = 256      # the dataset side: up to this many training pictures, spread evenly through the list
STYLE_TOP_K = 8             # a preview's score: the mean of its 8 closest training pictures (a varied set has no
                            # single centre)
STYLE_LO, STYLE_HI = 0.75, 0.95     # raw -> shown: calibrated on the box (29 Oct 2026, one artist's 240 pictures):
                                    # unrelated pictures ~0.78 raw (0.1 shown), other furry artists ~0.88 (0.64),
                                    # held-out pictures of the same artist ~0.93 (0.90)


def style_show(raw):
    return round(min(1.0, max(0.0, (float(raw) - STYLE_LO) / (STYLE_HI - STYLE_LO))), 4)


def style_images(dirs):
    """The training pictures the Style score compares with: the folders' pictures (no masks), sorted, capped."""
    from fizgig.dataset.masks import is_mask_file
    out = []
    for d in dirs:
        try:
            out += sorted(os.path.join(d, n) for n in os.listdir(d) if n.lower().endswith(PICS) and not is_mask_file(n))
        except OSError:
            continue
    if len(out) > STYLE_MAX_IMAGES:
        out = [out[round(i * (len(out) - 1) / (STYLE_MAX_IMAGES - 1))] for i in range(STYLE_MAX_IMAGES)]
    return out


def _style_bank(samples_dir, run, dirs, style_fn):
    """The training pictures' style vectors, worked out once and kept beside the previews (.style_<run>.npz; a new
    set of pictures works them out again)."""
    import hashlib
    import numpy as np
    imgs = style_images(dirs)
    if not imgs:
        return None
    key = hashlib.sha1("\n".join(imgs).encode("utf-8")).hexdigest()
    path = os.path.join(samples_dir, f".style_{run}.npz")
    try:
        with np.load(path) as z:
            if str(z["key"]) == key:
                return z["bank"]
    except (OSError, KeyError, ValueError):
        pass
    vecs = []
    for f in imgs:
        try:
            vecs.append(style_fn(f))
        except Exception:
            continue                   # an unreadable picture is left out
    if not vecs:
        return None
    bank = np.stack(vecs).astype("float32")
    try:
        np.savez(path, key=np.array(key), bank=bank)
    except OSError:
        pass
    return bank


def style_pass(samples_dir, style_fn=style_embed):
    """Style score every not-yet-scored preview of a run whose sidecar names its training folders: how close it is
    to the dataset's look as a whole (the mean of its STYLE_TOP_K closest training pictures), shown 0-1. Returns how
    many were added."""
    import numpy as np
    from fizgig.web import gallery_likeness as gl
    try:
        files = sorted(f for f in os.listdir(samples_dir) if f.lower().endswith(PICS))
    except OSError:
        return 0
    done = gl.read(samples_dir).get("style") or {}
    banks, new = {}, {}
    for f in files:
        m = _SAMPLE.match(f)
        if not m or f in done:
            continue
        run = m["run"]
        if run not in banks:
            dirs = read_style_dirs(samples_dir, run)
            banks[run] = _style_bank(samples_dir, run, dirs, style_fn) if dirs else None
        bank = banks[run]
        if bank is None:
            continue
        sims = np.sort(bank @ np.asarray(style_fn(os.path.join(samples_dir, f)), dtype="float32"))[::-1]
        raw = float(sims[:STYLE_TOP_K].mean())
        new[f] = {"score": style_show(raw), "raw": round(raw, 4)}
    if new:
        gl.merge_style(samples_dir, new, set(files))
    return len(new)


def score_pass(samples_dir, embed_fn=embed):
    """Score every not-yet-scored preview of a paired prompt in the folder; returns how many were added. A preview
    whose epoch recorded a different prompt at that index (a live prompt change) is skipped."""
    from fizgig.web import gallery_likeness as gl
    try:
        files = sorted(f for f in os.listdir(samples_dir) if f.lower().endswith(PICS))
    except OSError:
        return 0
    done = gl.read(samples_dir).get("paired") or {}
    side, rec, refs, new = {}, {}, {}, {}
    for f in files:
        m = _SAMPLE.match(f)
        if not m:
            continue
        run, ep, idx = m["run"], str(int(m["epoch"])), int(m["idx"])
        if run not in side:
            side[run] = {p["index"]: p for p in read_sidecar(samples_dir, run)}
            rec[run] = _epoch_prompts(samples_dir, run)
        p = side[run].get(idx)
        if not p or not os.path.exists(p.get("path") or ""):
            continue
        if (done.get(f) or {}).get("image") == os.path.basename(p["path"]):
            continue                    # scored against this image already (a re-pick rescores)
        said = rec[run].get(ep)
        if isinstance(said, list) and idx < len(said) and prompt_hash(said[idx]) != p["hash"]:
            continue
        if p["path"] not in refs:
            refs[p["path"]] = embed_fn(p["path"])
        e = embed_fn(os.path.join(samples_dir, f))
        new[f] = {"score": round(float(sum(a * b for a, b in zip(refs[p["path"]], e))), 4),
                  "image": os.path.basename(p["path"]), "index": idx}
    if new:
        gl.merge_paired(samples_dir, new, set(files))
    return len(new)


def update(samples_dir, run, prompts, pairs, dataset_dir, style_dirs=None):
    """Mid-run pairing (the Samples tab, no relaunch): rewrite .pairs_<run>.json from the form as it is now - the
    launch rule: written when something is paired, emptied when nothing is and a file exists - then score."""
    if not samples_dir or not run or os.path.basename(str(run)) != str(run) or not os.path.isdir(samples_dir):
        raise ValueError("No samples folder for this run yet.")
    text = sidecar_text(pairs, prompts, dataset_dir, style_dirs)
    n = len(json.loads(text)["pairs"])
    side = sidecar(samples_dir, run)
    if n or style_dirs or os.path.exists(side):
        tmp = side + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, side)
    if n or style_dirs:
        kick(samples_dir)
    return n


def kick(samples_dir):
    """One background pass (the gallery's poll calls this); nothing when a pass is already running for the folder."""
    if not samples_dir or not os.path.isdir(samples_dir) or not any(
            f.startswith(".pairs_") for f in os.listdir(samples_dir)):
        return
    with _lock:
        if samples_dir in _busy:
            return
        _busy.add(samples_dir)

    def go():
        try:
            score_pass(samples_dir)
            style_pass(samples_dir)
        except Exception as e:      # a broken model download must not take the gallery down
            print(f"[likeness] paired scoring stopped: {e}", flush=True)
        finally:
            with _lock:
                _busy.discard(samples_dir)
    threading.Thread(target=go, daemon=True).start()
