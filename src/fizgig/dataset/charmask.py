"""Character masks for drawn, anime and furry pictures: SkyTNT's anime-seg (Apache-2.0,
https://huggingface.co/skytnt/anime-seg, isnetis.onnx, 176 MB), run with onnxruntime (already a Fizgig requirement).
Google's person finder (automask.py) only knows real people: on a drawn fox or a fursuit it finds next to nothing.
This one cuts out the whole character - fur, tail, ears, wings - from drawings, anime, 3D renders and most fursuit
photos, as one shape (no hair / skin / clothes parts).

The masks card's "Who's in your pictures" picks the finder:
- "photo": Google's person finder, exactly as before (this file isn't used);
- "drawn": this character finder;
- "auto": both on every picture; a picture counts as a photo of a real person when the person finder sees human face
  skin and agrees with the character finder's outline (looks_like_photo), else the character finder's mask is used.

Parts for a character: the head comes from a YOLO head/face model (the face finder's YOLO model in Image Prep's
face settings - Fdetailer when it's there, else deepghs' MIT anime head model, downloaded on first use), so
"head", "not_faces" and "body_clothes" work; clothes can't be told from fur, so "no_clothes", "body" and "clothes"
stay photo-only (SAM 3.1 from words can do them: "shirt, jacket").

Downloaded once, on demand, from the masks card's Download button, its hash checked."""
import hashlib
import os
import threading
import time

import numpy as np

REPO = "skytnt/anime-seg"
FILE = "isnetis.onnx"
REVISION = "493cb60893f47441b26ec4fb9a306bce9e342982"
SHA256 = "f15622d853e8260172812b657053460e20806f04b9e05147d49af7bed31a6e99"
SIZE_MB = 176
FOLDER = "anime-seg"
LABEL = "SkyTNT anime-seg"
SIDE = 1024                                # the network's input: the picture fitted into a 1024 square

WHO = {"photo": "Real people (photos)",
       "drawn": "Drawn or furry characters",
       "auto": "A mix: decide for each picture"}
WHO_DEFAULT = "photo"
DRAWN_KINDS = ("person", "head", "body_clothes", "not_faces", "background")   # what a character mask can give
KIND_LABELS_DRAWN = {"person": "The whole character (fur, tail and all)",
                     "head": "Just the head",
                     "body_clothes": "The character without the head",
                     "not_faces": "Everything but the heads",
                     "background": "Just the background, not the characters"}
# Auto: a photo of a real person when the person finder's face skin covers at least this share of the picture...
PHOTO_FACE_SHARE = 0.003
# ...and at least this share of the character finder's outline is inside the person finder's (it sees the same body)
PHOTO_AGREE = 0.6
HEAD_GROW = 0.25                           # a head box grown by this much each way (ears, hair, a muzzle's tip)
# Close-ups (a face crop filling the frame): the character finder, trained on whole characters, sometimes sees nothing
# there. When it finds less than this share of the picture and covers less than CLOSEUP_HEAD_SEEN of the head finder's
# boxes, the (grown) head boxes count as the character.
CLOSEUP_MISS = 0.05
CLOSEUP_HEAD_SEEN = 0.25
# "Just the head" when the head finder sees no head: a character this big in the frame is a close-up of the head
CLOSEUP_HEAD_FILL = 0.5
NOTHING = 0.01                             # a character mask with less than this on average: nothing found
HEAD_KINDS = ("head", "body_clothes", "not_faces")

_lock = threading.Lock()
_fetch = {"state": "", "error": "", "at": 0.0}


def model_dir(models_dir):
    return os.path.join(models_dir, FOLDER)


def find_model(models_dir):
    """The downloaded isnetis.onnx, or "" before the download."""
    p = os.path.join(model_dir(models_dir), FILE)
    return p if os.path.isfile(p) else ""


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(models_dir):
    """Fetch the model into models/anime-seg on a worker thread (hash checked); fetch_state() follows it."""
    with _lock:
        if _fetch["state"] == "running":
            return
        _fetch.update(state="running", error="", at=time.time())

    def work():
        try:
            from huggingface_hub import hf_hub_download
            got = hf_hub_download(repo_id=REPO, filename=FILE, revision=REVISION, local_dir=model_dir(models_dir))
            if _sha256(got) != SHA256:
                os.remove(got)
                raise RuntimeError("the downloaded file isn't the expected one - try again later")
            with _lock:
                _fetch.update(state="done", at=time.time())
        except Exception as e:
            with _lock:
                _fetch.update(state="failed", error=f"{type(e).__name__}: {e}", at=time.time())
    threading.Thread(target=work, daemon=True).start()


def fetch_state(models_dir):
    """{"state", "error", "mb", "of"}: how the download is going (MB arrived, from hf_hub_download's part file)."""
    with _lock:
        st = dict(_fetch)
    mb = 0.0
    if st["state"] == "running":
        d = os.path.join(model_dir(models_dir), ".cache", "huggingface", "download")
        try:
            mb = max((os.path.getsize(os.path.join(d, n)) for n in os.listdir(d) if n.endswith(".incomplete")),
                     default=0) / 2 ** 20
        except OSError:
            pass
    return {"state": st["state"], "error": st["error"], "mb": round(mb), "of": SIZE_MB}


def info(models_dir):
    return {"path": find_model(models_dir), "fetch": fetch_state(models_dir), "mb": SIZE_MB, "label": LABEL}


def available():
    try:
        import cv2  # noqa: F401
        import onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


class CharacterFinder:
    """The loaded model: mask(img) -> (H, W) float32 in [0, 1], the character(s) at the picture's own size."""

    def __init__(self, path, session=None):
        if session is None:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.log_severity_level = 3
            avail = ort.get_available_providers()
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail] or avail
            session = ort.InferenceSession(path, sess_options=so, providers=providers)
        self.session = session
        self.input = session.get_inputs()[0].name

    def mask(self, img):
        import cv2
        a = np.asarray(img.convert("RGB"), np.float32) / 255.0
        h0, w0 = a.shape[:2]
        h, w = (SIDE, max(1, int(SIDE * w0 / h0))) if h0 > w0 else (max(1, int(SIDE * h0 / w0)), SIDE)
        top, left = (SIDE - h) // 2, (SIDE - w) // 2
        x = np.zeros((SIDE, SIDE, 3), np.float32)
        # shrinking: INTER_AREA; enlarging a small picture (a face crop) needs a smooth resize - INTER_AREA enlarges
        # in blocks of pixels and the network then sees no character at all (most crops under ~400 px came out empty)
        x[top:top + h, left:left + w] = np.clip(cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA if w <= w0 and
                                                           h <= h0 else cv2.INTER_LANCZOS4), 0.0, 1.0)
        y = np.asarray(self.session.run(None, {self.input: x.transpose(2, 0, 1)[None]})[0], np.float32)
        y = y.reshape(y.shape[-2], y.shape[-1])[top:top + h, left:left + w]
        return np.clip(cv2.resize(y, (w0, h0), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)

    def close(self):
        self.session = None


def looks_like_photo(classes, character):
    """Auto's choice for one picture: True for a photo of a real person (use the person finder's mask). classes: the
    person finder's (H, W, 6) probabilities; character: the character finder's (H, W) mask."""
    from fizgig.dataset.automask import BACKGROUND, FACE
    human = (1.0 - classes[..., BACKGROUND]) > 0.5
    char = character > 0.5
    face = float((classes[..., FACE] > 0.5).mean())
    if float(character.mean()) < NOTHING:
        return face >= PHOTO_FACE_SHARE       # the character finder saw nothing (a close-up photo): human face skin decides
    agree = float((human & char).sum()) / max(1, int(char.sum()))
    return face >= PHOTO_FACE_SHARE and agree >= PHOTO_AGREE


def head_region(shape, boxes, grow=HEAD_GROW):
    """(H, W) bool: the head boxes (x1, y1, x2, y2), each grown by `grow` of its size each way."""
    h, w = shape
    out = np.zeros((h, w), bool)
    for x1, y1, x2, y2 in boxes:
        gx, gy = (x2 - x1) * grow, (y2 - y1) * grow
        a, b = max(0, int(x1 - gx)), max(0, int(y1 - gy))
        c, d = min(w, int(np.ceil(x2 + gx))), min(h, int(np.ceil(y2 + gy)))
        if c > a and d > b:
            out[b:d, a:c] = True
    return out


def mask_from(character, kind, heads=None):
    """The mask (H, W) float in [0, 1] of a kind for a drawn character. heads: head_region() (needed for "head",
    "body_clothes" and "not_faces")."""
    if kind not in DRAWN_KINDS:
        raise ValueError(f"{kind!r} needs a photo of a real person")
    if kind == "person":
        m = character
    elif kind == "background":
        m = 1.0 - character
    else:
        if heads is None:
            raise ValueError(f"{kind!r} needs the head finder")
        if kind == "head":
            m = character * heads
        elif kind == "body_clothes":
            m = character * ~heads
        else:                                                 # not_faces
            m = 1.0 - character * heads
    return np.clip(m, 0.0, 1.0).astype(np.float32)


def drawn_mask(character, kind, find_heads=None):
    """A drawn character's mask of a kind, with the close-up fallbacks: (mask (H, W) float32 or None when there's
    nothing to mask, a note, the character mask used). find_heads() -> head boxes (x1, y1, x2, y2), asked only when needed (None: no head
    finder). The note says what was done ("" normally) or, with None, why nothing was found."""
    if kind not in DRAWN_KINDS:
        return None, "a drawn or furry character - can't tell clothes from fur", character
    boxes = None

    def heads():
        nonlocal boxes
        if boxes is None:
            boxes = list(find_heads() or []) if find_heads is not None else []
        return boxes
    note = ""
    if float(character.mean()) < CLOSEUP_MISS and heads():
        hr = head_region(character.shape, boxes)
        if float(character[hr].mean()) < CLOSEUP_HEAD_SEEN:   # the finder missed the head the head finder sees
            character = np.maximum(character, hr.astype(np.float32))
            note = "a close-up: the head finder's box used"
    if float(character.mean()) < NOTHING:
        return None, ("no character found (the head finder found no head either)" if find_heads is not None and
                      not boxes else "no character found"), character
    hr = None
    if kind in HEAD_KINDS:
        if heads():
            hr = head_region(character.shape, boxes)
        elif float((character > 0.5).mean()) >= CLOSEUP_HEAD_FILL:
            hr = np.ones(character.shape, bool)               # a character filling the frame: a close-up of the head
            note = "a close-up: the whole character counted as the head"
        else:
            hr = np.zeros(character.shape, bool)
    m = mask_from(character, kind, hr)
    from fizgig.dataset.masks import MIN_COVER
    if float(m.mean()) < MIN_COVER:
        why = {"head": "a character but no head found",
               "body_clothes": "only a head here, nothing else to mask",
               "not_faces": "only a head here, nothing else to mask",
               "background": "the character fills the picture: no background to mask"}.get(kind, "nothing found")
        return None, why, character
    return m, note, character
