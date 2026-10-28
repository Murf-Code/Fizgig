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


def sidecar_text(pairs, prompts, dataset_dir):
    """What launch writes: the resolved pairings with the image's full path."""
    ok, _ = resolve(pairs, prompts)
    for p in ok:
        p["path"] = os.path.join(dataset_dir or "", *dataset_rel(p["image"]).split("/"))
    return json.dumps({"pairs": ok})


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


def update(samples_dir, run, prompts, pairs, dataset_dir):
    """Mid-run pairing (the Samples tab, no relaunch): rewrite .pairs_<run>.json from the form as it is now - the
    launch rule: written when something is paired, emptied when nothing is and a file exists - then score."""
    if not samples_dir or not run or os.path.basename(str(run)) != str(run) or not os.path.isdir(samples_dir):
        raise ValueError("No samples folder for this run yet.")
    text = sidecar_text(pairs, prompts, dataset_dir)
    n = len(json.loads(text)["pairs"])
    side = sidecar(samples_dir, run)
    if n or os.path.exists(side):
        tmp = side + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, side)
    if n:
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
        except Exception as e:      # a broken model download must not take the gallery down
            print(f"[likeness] paired scoring stopped: {e}", flush=True)
        finally:
            with _lock:
                _busy.discard(samples_dir)
    threading.Thread(target=go, daemon=True).start()
