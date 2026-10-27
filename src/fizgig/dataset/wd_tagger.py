"""SmilingWolf's WD EVA02-Large tagger v3 (Apache-2.0, https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3):
Image Prep's "Trim tags that don't fit" asks it which of a crop's copied tags it still sees. Downloaded once, on
demand, into Fizgig's models folder (model.onnx + selected_tags.csv, about 1.2 GB); run with onnxruntime (already a
Fizgig requirement) on the GPU when its CUDA provider is there, else the CPU; freed when the prep run ends.

Only general tags (category 0) are ever dropped: character (4), rating (9), meta and any tag the tagger doesn't know
(species it lacks, artists, your own tags) always stay. Tags are only removed, never added."""
import csv
import os
import threading
import time

from fizgig.utils import hf_public

REPO = "SmilingWolf/wd-eva02-large-tagger-v3"
FILES = ("model.onnx", "selected_tags.csv")
FOLDER = "wd-eva02-large-tagger-v3"
LABEL = "WD EVA02-Large tagger v3"
SIZE_GB = 1.2
GENERAL = 0
KEEP_SCORE = 0.35                        # a general tag stays if the tagger scores it at least this on the crop

_lock = threading.Lock()
_fetch = {"state": "", "error": "", "at": 0.0}


def norm(tag):
    """'Long_Hair' / 'long hair' / 'fox \\(species\\)' -> 'long hair' / 'fox (species)': how a caption's tag and the
    tagger's name are compared."""
    t = str(tag).strip().lower().replace("\\(", "(").replace("\\)", ")")
    if len(t) > 3:                       # kaomoji like ^_^ keep their underscores
        t = t.replace("_", " ")
    return " ".join(t.split())


def model_dir(models_dir):
    return os.path.join(models_dir, FOLDER)


def find_model(models_dir):
    """The tagger's folder when both files are there, else ""."""
    d = model_dir(models_dir)
    return d if all(os.path.isfile(os.path.join(d, f)) for f in FILES) else ""


def download(models_dir):
    """Fetch the two files into models/wd-eva02-large-tagger-v3 on a worker thread; fetch_state() follows it."""
    with _lock:
        if _fetch["state"] == "running":
            return
        _fetch.update(state="running", error="", at=time.time())

    def work():
        try:
            for f in FILES[::-1]:                     # the small tag list first
                hf_public.download(repo_id=REPO, filename=f, local_dir=model_dir(models_dir))
            with _lock:
                _fetch.update(state="done", at=time.time())
        except Exception as e:
            with _lock:
                _fetch.update(state="failed", error=hf_public.friendly_error(e), at=time.time())
    threading.Thread(target=work, daemon=True).start()


def fetch_state(models_dir):
    with _lock:
        st = dict(_fetch)
    gb = 0.0
    if st["state"] == "running":
        d = os.path.join(model_dir(models_dir), ".cache", "huggingface", "download")
        try:
            gb = max((os.path.getsize(os.path.join(d, n)) for n in os.listdir(d) if n.endswith(".incomplete")),
                     default=0) / 2 ** 30
        except OSError:
            pass
    return {"state": st["state"], "error": st["error"], "gb": round(gb, 2), "of": SIZE_GB}


def info(models_dir):
    return {"path": find_model(models_dir), "fetch": fetch_state(models_dir), "gb": SIZE_GB, "label": LABEL}


def read_tags(path):
    """selected_tags.csv -> [(name, category)] in the model's output order."""
    with open(path, encoding="utf-8", newline="") as f:
        return [(row["name"], int(row["category"])) for row in csv.DictReader(f)]


def preprocess(img, size):
    """A PIL picture as the WD v3 taggers take it: on white, padded square, resized to size, BGR, float32 0-255,
    (1, size, size, 3)."""
    import numpy as np
    from PIL import Image
    img = img.convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(bg, img).convert("RGB")
    side = max(img.size)
    sq = Image.new("RGB", (side, side), (255, 255, 255))
    sq.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    if side != size:
        sq = sq.resize((size, size), Image.BICUBIC)
    a = np.asarray(sq, dtype=np.float32)[:, :, ::-1]
    return np.ascontiguousarray(a)[None]


class Tagger:
    """The loaded tagger: scores(img) -> {normalised tag: probability}; categories: {normalised tag: category}."""

    def __init__(self, folder, log=print, session=None):
        self.tags = read_tags(os.path.join(folder, "selected_tags.csv"))
        self.categories = {norm(n): c for n, c in self.tags}
        if session is None:
            import onnxruntime as ort
            avail = ort.get_available_providers()
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail] or avail
            log(f"Loading {LABEL} ({'GPU' if providers[0].startswith('CUDA') else 'CPU'}) ...\n")
            session = ort.InferenceSession(os.path.join(folder, "model.onnx"), providers=providers)
        self.session = session
        inp = self.session.get_inputs()[0]
        self.input = inp.name
        self.size = int(inp.shape[1]) if isinstance(inp.shape[1], int) else 448

    def scores(self, img):
        x = preprocess(img, self.size)
        probs = self.session.run(None, {self.input: x})[0][0]
        return {norm(n): float(p) for (n, _c), p in zip(self.tags, probs)}

    def close(self):
        self.session = None
        import gc
        gc.collect()


def trim(caption, scores, categories, keep_words=(), keep_score=KEEP_SCORE):
    """(caption, dropped): a tag caption without the general tags the tagger no longer sees on the crop (scored
    under keep_score). The first tag, the trigger word, non-general tags and tags the tagger doesn't know stay;
    a sentence caption comes back as written."""
    from fizgig.dataset.tag_variants import split_tags
    tags = split_tags(caption)
    if tags is None:
        return caption, []
    keep = {norm(w) for w in keep_words if w and str(w).strip()}
    kept, dropped = [], []
    for i, t in enumerate(tags):
        n = norm(t)
        if i == 0 or n in keep or categories.get(n) != GENERAL or scores.get(n, 0.0) >= keep_score:
            kept.append(t)
        else:
            dropped.append(t)
    return (", ".join(kept), dropped) if dropped else (caption, [])
