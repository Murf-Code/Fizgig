"""The web UI's Image Prep: a prep run (resize / PNG / face close-ups) and the Look Consistency Filter's scan, each
in a background thread on the Fizgig machine, with their state for the page to poll. The work itself is
fizgig/dataset/prep.py - the same code the desktop app's Image Prep tab runs. Face tools run on the CPU, with the face
finder and look match Image Prep's face settings pick (face_utils: InsightFace / YOLO / Auto, ArcFace / DINOv2).
"""
import base64
import io
import json
import os
import threading
import time

from fizgig.dataset import prep
from fizgig.utils.upright import upright

_lock = threading.Lock()
_job = {"state": "idle", "log": "", "folder": "", "started": 0.0}
_look = {"state": "idle", "folder": "", "baselines": [], "scores": {}, "progress": [0, 0], "error": "",
         "embedder": "", "thresholds": None, "decision": ""}
_embed_cache = {}             # (path, mtime, embedder settings) -> embedding or None
_detector = None
_embedders = {}               # FaceEmbedder.settings_key -> FaceEmbedder (one per look match in use)
_model_lock = threading.Lock()


def faces_available() -> bool:
    """Whether the face finder Image Prep's face settings pick can run here (the default, InsightFace: InsightFace,
    OpenCV and onnxruntime installed, as before)."""
    try:
        from fizgig.face_utils import FaceDetector
        return bool(FaceDetector().available)
    except Exception:
        return False


def _face_detector():
    """The face finder for the current face settings; made again when they change."""
    global _detector
    from fizgig.face_utils import FaceDetector
    with _model_lock:
        fresh = FaceDetector()
        if _detector is None or _detector.settings_key != fresh.settings_key:
            _detector = fresh
    return _detector


def face_embedder(settings=None):
    """The FaceEmbedder for these face settings (default: prefs.json), one per look match: the model loads once."""
    from fizgig.face_utils import FaceEmbedder
    fresh = FaceEmbedder(settings=settings)
    with _model_lock:
        return _embedders.setdefault(fresh.settings_key, fresh)


def session_embedder(paths, log=None):
    """(embedder, decision message) for one scoring session over these pictures: with Auto look match, ArcFace for a
    human set and DINOv2 for a furry / anime / mixed one (face_utils.decide_look_match) - never a mix of the two."""
    from fizgig.face_utils import decide_look_match
    d = decide_look_match(list(paths), log=log)
    return face_embedder(d.settings), ("" if d.kind == "explicit" else d.message)   # only Auto's choice is news


FACE_DETECTOR_LABELS = {"insightface": "People (InsightFace, the default)",
                        "yolo": "Anime/furry/drawn (YOLO model)",
                        "auto": "Both, for sets with people and characters"}
FACE_IDENTITY_LABELS = {"arcface": "People (the default)", "dinov2": "Anime/furry/drawn",
                        "auto": "Auto (Decide for each set)"}


def face_settings():
    """Image Prep's face settings for the page: the saved values (prefs.json face_*), the choices, and what they come
    to (the YOLO model actually used, whether the finder can tell gender)."""
    from fizgig import face_utils as F
    st = F.load_face_settings()
    try:
        with open(F.PREFS_FILE, encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, ValueError):
        saved = {}
    det = F.FaceDetector(settings=st)
    return {"detector": st["face_detector"], "identity": st["face_identity"],
            "yolo_model": str(saved.get("face_yolo_model") or ""), "auto_threshold": st["face_auto_threshold"],
            "detectors": [[k, FACE_DETECTOR_LABELS[k]] for k in ("insightface", "yolo", "auto")],
            "identities": [[k, FACE_IDENTITY_LABELS[k]] for k in ("arcface", "dinov2", "auto")],
            "defaults": {"detector": F.FACE_SETTING_DEFAULTS["face_detector"],
                         "identity": F.FACE_SETTING_DEFAULTS["face_identity"]},
            "finder": det.describe(), "has_gender": det.has_gender,
            "insightface": F.is_insightface_available(), "local_model": F.LOCAL_YOLO_DEFAULT,
            **_class_info()}


def _class_info():
    """face_settings()'s YOLO part: Crop to's extra classes, and every class name (Find this phrases the model finds
    itself, so SAM 3.1 isn't needed for them)."""
    b = _yolo()
    try:
        names = list(b.all_classes().values()) if b is not None else []
    except Exception:
        names = []
    return {"classes": yolo_classes(b) if b is not None else [], "yolo_names": names}


def save_face_settings(b):
    """Check and save the face settings sent by the page (only the keys given); the face finder and the look match
    pick them up on their next use."""
    from fizgig import face_utils as F
    ch = {}
    if "detector" in b:
        if b["detector"] not in F.DETECTORS:
            raise ValueError("Unknown face finder.")
        ch["face_detector"] = b["detector"]
    if "identity" in b:
        if b["identity"] not in F.IDENTITIES:
            raise ValueError("Unknown look match.")
        ch["face_identity"] = b["identity"]
    if "yolo_model" in b:
        m = str(b["yolo_model"] or "").strip().strip('"')
        full = m if not m or os.path.isabs(m) else os.path.join(F._HERE, m)
        if m and not (full.lower().endswith(".onnx") and os.path.isfile(full)):
            raise ValueError("Pick an .onnx face model on the Fizgig computer. docs/FACE_BACKENDS.md explains how to "
                             "make one from a .pt file.")
        ch["face_yolo_model"] = m
    if "auto_threshold" in b:
        try:
            v = float(b["auto_threshold"])
        except (TypeError, ValueError):
            raise ValueError("The Auto threshold is a number from 0 to 1.")
        if not 0.0 <= v <= 1.0:
            raise ValueError("The Auto threshold is a number from 0 to 1.")
        ch["face_auto_threshold"] = v
    return ch


def busy() -> bool:
    with _lock:
        return _job["state"] == "running" or _look["state"] == "running"


# ---- a prep run -------------------------------------------------------------------------------------------------
def status(since=0):
    with _lock:
        log = _job["log"]
        return {"state": _job["state"], "folder": _job["folder"], "log": log[since:], "size": len(log)}


def _log(text):
    with _lock:
        _job["log"] = (_job["log"] + text)[-400_000:]


def _yolo():
    """The YOLO face model when it is the face finder (face_utils.yolo_backend), loaded, or None."""
    from fizgig import face_utils as F
    return F.yolo_backend()


def yolo_classes(backend=None):
    """Crop to's extra choices: the YOLO face model's other classes ([{value, label, name}]; [] for a face-only model,
    or when the face finder isn't the YOLO one)."""
    from fizgig import face_utils as F
    b = backend if backend is not None else _yolo()
    if b is None:
        return []
    try:
        extra = b.extra_classes()
    except Exception:
        return []
    return [{"value": prep.class_target(n), "label": F.class_label(n), "name": n} for _k, n in extra.items()]


def yolo_words(words, backend=None):
    """{phrase: class id} for each phrase that names one of the YOLO face model's classes (case, spaces, _ and -
    ignored: 'paw pads' is 'pawpads'). {} when the face finder isn't the YOLO one."""
    from fizgig import face_utils as F
    b = backend if backend is not None else _yolo()
    if b is None or not words:
        return {}
    try:
        names = b.all_classes()
    except Exception:
        return {}
    keys = {F.class_key(n): k for k, n in names.items()}
    keys.update({F.class_key(F.class_label(n)): k for k, n in names.items()})
    return {w: keys[F.class_key(w)] for w in words if F.class_key(w) in keys}


def find_words(face_label, find, backend=None):
    """With the Face dropdown on Other… or one of the YOLO model's classes: (the phrases to find, the SAM 3.1 file),
    checked - ValueError with a plain note when the box is empty, or SAM 3.1 is needed (a phrase that isn't one of
    the YOLO model's classes) and isn't downloaded. The SAM file is "" when the YOLO model finds every phrase.
    ([], "") for a face choice."""
    from fizgig.dataset import sammask
    from fizgig.user_data import MODELS_DIR
    cls = prep.class_of(face_label)
    if cls:
        b = backend if backend is not None else _yolo()
        if not yolo_words([cls], b):
            raise ValueError(f"The face model doesn't find {cls}: pick the YOLO face model that has it under "
                             "Anime/furry/drawn characters, or pick another Crop to.")
        return [cls], ""
    if not prep.is_other(face_label):
        return [], ""
    words = sammask.split_words(find)
    if not words:
        raise ValueError("Type what to find in Find this (e.g. paw, tail, hand).")
    by_yolo = yolo_words(words, backend)
    if all(w in by_yolo for w in words):
        return words, ""
    path = sammask.find_model(MODELS_DIR)
    if not path:
        raise ValueError("Finding by words needs SAM 3.1: press Download on the Masks card (further down this tab) "
                         "first.")
    return words, path


def thing_finder(words, sam_path, log=print, backend=None):
    """The ThingFinder for these phrases: the YOLO face model for the ones it has a class for, SAM 3.1 for the rest
    (loaded only when some phrase needs it). The caller holds sammask._use when sam_path is set."""
    from fizgig.dataset import sammask
    b = backend if backend is not None else _yolo()
    by_yolo = yolo_words(words, b)
    seg = sammask.segmenter(sam_path, log) if sam_path and any(w not in by_yolo for w in words) else None
    return prep.ThingFinder(seg, words, yolo=b if by_yolo else None, yolo_words=by_yolo)


def keep_score(v, default=None):
    """Trim's keep score from the page: a number from 0 to 1 (ValueError otherwise); blank is the tagger's default
    (`default`, else the WD tagger's)."""
    from fizgig.dataset import wd_tagger
    if v is None or str(v).strip() == "":
        return wd_tagger.KEEP_SCORE if default is None else default
    try:
        f = float(v)
    except (TypeError, ValueError):
        f = -1.0
    if not 0.0 < f < 1.0:
        raise ValueError("Keep tags scored at least: a number between 0 and 1, like 0.35.")
    return f


def caption_settings(c):
    """The page's caption settings for crops ({copy, trim, tagger, keep_score, ext, trigger}), checked: (settings or
    None, the tagger's folder or ""). tagger is "wd-eva02" (the default) or "jtp3" (taggers.py). ValueError when Trim
    is on and that tagger isn't downloaded."""
    if not isinstance(c, dict):
        return None, ""
    from fizgig.dataset import taggers
    from fizgig.user_data import MODELS_DIR
    from fizgig.web.captions import caption_ext
    name = taggers.pick(c.get("tagger"))
    mod = taggers.get(name)
    c = {"copy": c.get("copy", True) is not False, "trim": bool(c.get("trim")), "tagger": name,
         "keep_score": keep_score(c.get("keep_score"), mod.KEEP_SCORE), "ext": caption_ext(c.get("ext")),
         "trigger": str(c.get("trigger") or "").strip()}
    folder = ""
    if c["copy"] and c["trim"]:
        folder = mod.find_model(MODELS_DIR)
        if not folder:
            raise ValueError(f"Trim tags that don't fit needs the {mod.LABEL}: press Download under it first, "
                             "or untick Trim.")
    return c, folder


def crop_captions(c, tagger=None):
    """Checked caption settings (caption_settings) as a CropCaptions, trimming with `tagger`; None for none."""
    if c is None:
        return None
    from fizgig.dataset.crop_captions import CropCaptions
    return CropCaptions(copy=c["copy"], ext=c["ext"], keep=[c["trigger"]], tagger=tagger if c["trim"] else None,
                        keep_score=c["keep_score"])


def _tagger(folder, name=None):
    from fizgig.dataset import taggers
    return taggers.get(name).Tagger(folder, log=_log)


def start(folder, mode, mp, face_label, padding, replace, find="", captions=None):
    if not folder or not os.path.isdir(folder):
        raise ValueError("Choose the training folder on the Start tab first.")
    if mode not in prep.MODES:
        raise ValueError("Pick what to do.")
    words, sam_path = find_words(face_label, find) if mode != "Resize Only" else ([], "")
    cap_settings, tagger_dir = caption_settings(captions) if mode != "Resize Only" else (None, "")
    if mode != "Resize Only" and not words and not faces_available():
        raise ValueError("Face detection isn't installed on this machine: run install_fizgig.py to add it.")
    with _lock:
        if _job["state"] == "running":
            raise ValueError("A prep run is already going.")
        if _look["state"] == "running":
            raise ValueError("The Look Filter is scanning. Wait for it to finish.")
        _job.update(state="running", log="", folder=folder, started=time.time())

    def work():
        try:
            tagger = _tagger(tagger_dir, cap_settings["tagger"]) if tagger_dir else None
            try:
                caps = crop_captions(cap_settings, tagger)
                if words:
                    _things(words, sam_path, mode, folder, mp, face_label, padding, replace, caps)
                else:
                    det = _face_detector() if mode != "Resize Only" else None
                    prep.Prep(_log, det, captions=caps).run(mode, folder, float(mp), face_label, float(padding),
                                                            bool(replace))
            finally:
                if tagger is not None:
                    tagger.close()                        # the tagger is freed when the run ends
            _log("\nDone.\n")
            end = "done"
        except Exception as e:
            _log(f"\nERROR: prep failed — {type(e).__name__}: {e}\n")
            end = "failed"
        with _lock:
            _job["state"] = end
    threading.Thread(target=work, daemon=True).start()
    return {"ok": True}


def _things(words, sam_path, mode, folder, mp, face_label, padding, replace, caps=None):
    """A prep run with Other… or a YOLO class: the YOLO face model finds its classes, SAM 3.1 the other words; SAM is
    freed from memory when the run ends."""
    from fizgig.dataset import sammask
    if not sam_path:                                   # the YOLO model finds it all: SAM 3.1 isn't touched
        finder = thing_finder(words, "", _log)
        prep.Prep(_log, None, finder, caps).run(mode, folder, float(mp), face_label, float(padding), bool(replace))
        return
    if not sammask._use.acquire(blocking=False):
        raise ValueError("SAM 3.1 is busy with a mask preview. Try again in a moment.")
    try:
        finder = thing_finder(words, sam_path, _log)
        prep.Prep(_log, None, finder, caps).run(mode, folder, float(mp), face_label, float(padding), bool(replace))
    finally:
        sammask._use.release()
        sammask.unload()


def test_things(path, words, sam_path):
    """Test Find this (or a YOLO class) on one photo: the picture with what was found boxed and labelled with its
    name and score (the one each phrase crops to green), writes nothing."""
    from PIL import Image, ImageDraw
    from fizgig.dataset import sammask
    if not os.path.isfile(path):
        raise ValueError("That picture isn't there.")
    with Image.open(path) as img:
        img = upright(img).convert("RGB")
    if sam_path:
        if not sammask._use.acquire(blocking=False):
            raise ValueError("SAM 3.1 is busy. Try again in a moment.")
        try:
            finder = thing_finder(words, sam_path)
            hits = finder.find(img)
        finally:
            sammask._use.release()
    else:
        finder = thing_finder(words, "")
        hits = finder.find(img)
    if not hits:
        return {"faces": [], "image": None, "finder": finder.describe(), "things": True}
    picked = finder.per_word(hits)
    shown = img.copy()
    d = ImageDraw.Draw(shown)
    lw = max(2, min(shown.size) // 250)
    for h in hits:
        col = (40, 200, 70) if h in picked else (240, 200, 40)
        d.rectangle(h.bbox, outline=col, width=lw)
        d.text((h.bbox[0] + lw + 2, h.bbox[1] + lw + 2), f"{h.gender} {h.score:.2f}", fill=col)
    shown.thumbnail((1400, 1400))
    buf = io.BytesIO()
    shown.save(buf, "JPEG", quality=88)
    return {"image": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii"),
            "finder": finder.describe(), "things": True,
            "faces": [{"gender": h.gender, "area": int(h.area), "selected": h in picked, "source": h.source,
                       "score": round(h.score, 2)} for h in hits]}


def test_faces(path, face_label):
    """Test face detection on one photo: the picture with the faces boxed (the chosen one green) as a JPEG data URL,
    and each face found. Writes nothing."""
    from PIL import Image
    from fizgig.face_utils import draw_face_boxes
    if not os.path.isfile(path):
        raise ValueError("That picture isn't there.")
    det = _face_detector()
    faces = det.detect_all(path)
    if not faces:
        return {"faces": [], "image": None, "finder": det.describe()}
    job = prep.Prep(lambda _t: None, det)
    sel, _ = job.select_face(faces, prep.face_mode_of(face_label))
    hi = faces.index(sel) if sel in faces else None
    with Image.open(path) as img:
        img = upright(img)
        img = img.convert("RGB") if img.mode not in ("RGB", "RGBA") else img
        shown = draw_face_boxes(img, faces, highlight_index=hi).convert("RGB")
    shown.thumbnail((1400, 1400))
    buf = io.BytesIO()
    shown.save(buf, "JPEG", quality=88)
    return {"image": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii"),
            "finder": det.describe(),
            "faces": [{"gender": f.gender, "area": int(f.area), "selected": i == hi,
                       "source": getattr(f, "source", "insightface")} for i, f in enumerate(faces)]}


# ---- the Look Consistency Filter --------------------------------------------------------------------------------
def _embed(path, embedder=None):
    """A face embedding through the (path, mtime, embedder) cache: re-scans and baseline swaps cost almost nothing.
    embedder: a session's (session_embedder); default the face settings' own."""
    emb = embedder or face_embedder()
    try:
        key = (path, os.path.getmtime(path), emb.settings_key)
    except OSError:
        return None
    if key not in _embed_cache:
        try:
            v = emb.embed(path)
        except Exception as e:
            print(f"[faces] {os.path.basename(path)}: {emb.label} failed ({type(e).__name__}: {e})", flush=True)
            v = None
        with _model_lock:
            _embed_cache[key] = v
    return _embed_cache[key]


def embed_face_box(path, embedder=None):
    """(embedding, bbox) for the largest face in a picture file - the desktop's _repair_embed_pil - on the session's
    embedder (default the face settings' own). The box is None when only the padded retry found the face (its box is
    in padded space). (None, None) without a face."""
    from PIL import Image
    emb = embedder or face_embedder()
    try:
        if emb.name == "arcface":       # read as before (OpenCV, unicode-safe): the same pixels, the same numbers
            import cv2
            import numpy as np
            bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            return (None, None) if bgr is None else emb.embed_bgr(bgr)
        with Image.open(path) as img:
            return emb.embed_pil(upright(img).convert("RGB"))
    except Exception:
        return None, None


def look_status():
    with _lock:
        scores = dict(_look["scores"])
        base = list(_look["baselines"])
        state, err, prog, folder = _look["state"], _look["error"], list(_look["progress"]), _look["folder"]
        thr, emb_label, decision = _look["thresholds"], _look["embedder"], _look["decision"]
    floor = (thr or prep.ARCFACE)["floor"]
    med, cut = prep.cutoff(scores, floor) if scores else (None, None)
    rows = []
    for p, s in scores.items():
        label, colour, blurb = prep.verdict(s, thr)
        rows.append({"path": p, "name": os.path.basename(p), "score": s, "verdict": label, "colour": colour,
                     "blurb": blurb, "baseline": p in base, "suggest": bool(cut is not None and s is not None
                                                                          and s < cut and p not in base)})
    rows.sort(key=lambda r: (r["score"] is None, r["score"] if r["score"] is not None else 0.0))   # worst first
    return {"state": state, "error": err, "progress": prog, "folder": folder, "baselines": base, "rows": rows,
            "median": med, "cutoff": cut, "embedder": emb_label, "decision": decision,
            "thresholds": thr or prep.ARCFACE}


def look_scan(folder, baselines):
    if not folder or not os.path.isdir(folder):
        raise ValueError("Choose the training folder on the Start tab first.")
    if not faces_available():
        raise ValueError("Face detection isn't installed on this machine: run install_fizgig.py to add it.")
    base = [os.path.normpath(b) for b in baselines or []]
    if len(base) != 3 or not all(os.path.isfile(b) for b in base):
        raise ValueError("Pick exactly 3 baseline pictures.")
    files = prep.look_files(folder)
    if not files:
        raise ValueError("No pictures in the training folder.")
    with _lock:
        if _look["state"] == "running":
            raise ValueError("A scan is already going.")
        if _job["state"] == "running":
            raise ValueError("Image prep is running. Wait for it to finish.")
        _look.update(state="running", folder=folder, baselines=base, scores={}, progress=[0, len(files)], error="",
                     embedder="", thresholds=None, decision="")

    def progress(i, total):
        with _lock:
            _look["progress"] = [i, total]

    def work():
        try:
            # one look match for the whole folder (Auto: decided from a sample of it), shown with the scores
            emb, msg = session_embedder(files, log=lambda m: print(m, flush=True))
            with _lock:
                _look.update(embedder=emb.label, thresholds=dict(emb.thresholds), decision=msg)
            scores = prep.score(files, base, lambda p: _embed(p, emb), progress)
            try:
                prep.save_scores(folder, base, scores, emb)     # for training's "Warm up look outliers"
            except OSError:
                pass
            with _lock:
                _look.update(state="done", scores=scores)
        except Exception as e:
            with _lock:
                _look.update(state="failed", error=str(e))
    threading.Thread(target=work, daemon=True).start()
    return {"ok": True}


def look_move(folder, paths, caption_ext=".txt"):
    """Move the marked pictures (and their captions) to excluded_by_look/; they leave the scan's list."""
    with _lock:
        if _look["state"] == "running":
            raise ValueError("A scan is going. Wait for it to finish.")
        known = set(_look["scores"])
    paths = [os.path.normpath(p) for p in paths or [] if os.path.normpath(p) in known]   # a marked baseline goes too
    moved, errors = prep.move_out(folder, paths, (".txt", caption_ext or ".txt"))
    with _lock:
        for p in moved:
            _look["scores"].pop(p, None)
    return {"moved": len(moved), "errors": errors, "dest": os.path.join(folder, "excluded_by_look"),
            "left": len(_look["scores"])}


# ---- Make masks (masked training) -------------------------------------------------------------------------------
def masks_available() -> bool:
    from fizgig.dataset import automask
    return automask.available()


def look_baselines(folder):
    """The Look Filter's three baselines saved with this folder (fizgig_look_scores.json), as paths; [] without."""
    try:
        with open(os.path.join(folder, "fizgig_look_scores.json"), encoding="utf-8") as f:
            names = json.load(f).get("baselines") or []
    except (OSError, ValueError):
        return []
    paths = [os.path.join(folder, n) for n in names]
    return paths if len(paths) == 3 and all(os.path.isfile(p) for p in paths) else []


def quick_check(kind, who):
    """Why the quick finder can't make these masks here (a message), or "" when it can (who: charmask.WHO)."""
    from fizgig.dataset import automask, charmask
    from fizgig.user_data import MODELS_DIR
    if who not in charmask.WHO:
        return "Pick who's in your pictures."
    if kind not in automask.KINDS:
        return "Pick what the masks should show."
    if who == "drawn" and kind not in charmask.DRAWN_KINDS:
        return "That one needs photos of real people: the character finder can't tell clothes from fur."
    if who != "drawn" and not automask.available():
        return "Making masks needs the ai-edge-litert package: run the updater (update_fizgig) to add it."
    if who != "photo" and not charmask.available():
        return "The character finder needs onnxruntime: run the updater (update_fizgig) to add it."
    if who != "photo" and not charmask.find_model(MODELS_DIR):
        return "The character finder isn't here yet: press Download on the masks card."
    return ""


def _head_finder():
    """The head finder for a drawn character's head kinds: the face settings' YOLO model (Fdetailer when it's there,
    else deepghs' anime head model, downloaded on first use)."""
    from fizgig.face_utils import FaceDetector
    det = FaceDetector(backend="yolo")
    if not det.available:
        raise ValueError("The head finder (a YOLO model) can't run here: see the face settings on this tab.")
    return det


class _LazyHeads:
    """The head finder for close-ups the character finder misses (charmask.drawn_mask), loaded on first use; where
    it can't run, no heads (the picture just gets no mask, as before)."""

    def __init__(self):
        self._det = None

    def detect_from_pil(self, img):
        if self._det is None:
            try:
                self._det = _head_finder()
            except Exception:
                self._det = False
        try:
            return self._det.detect_from_pil(img) if self._det else []
        except Exception:
            return []


def _heads_for(kind):
    """A drawn character's head finder: needed now for the head kinds, else only for close-ups (loaded if asked)."""
    from fizgig.dataset import charmask
    return _head_finder() if kind in charmask.HEAD_KINDS else _LazyHeads()


def start_masks(folder, kind, subject_only=False, overwrite=False, sam=None, who="photo"):
    """Masks for every picture in the training folder (dataset/automask.py), as a prep run: the same log and state.
    sam: {"path", "include", "exclude", "drawn"} to make them with SAM 3.1 from words instead (dataset/sammask.py).
    who: the quick finder's "Who's in your pictures" (dataset/charmask.py WHO)."""
    from fizgig.dataset import automask, charmask
    from fizgig.user_data import MODELS_DIR
    if not folder or not os.path.isdir(folder):
        raise ValueError("Choose the training folder on the Start tab first.")
    if sam is not None:
        if not sam.get("path"):
            raise ValueError("SAM 3.1 isn't here yet: press Download on the masks card.")
        if not sam.get("include"):
            raise ValueError("Type what the masks should show in Mask these (e.g. person).")
    else:
        why = quick_check(kind, who)
        if why:
            raise ValueError(why)
    bases = look_baselines(folder) if subject_only else []
    if subject_only and not bases:
        raise ValueError("Keeping only your subject needs the Look Filter's 3 baselines: choose them and scan the "
                         "folder (the last card on this tab) first.")
    if subject_only and not faces_available():
        raise ValueError("Face detection isn't installed on this machine: run install_fizgig.py to add it.")
    with _lock:
        if _job["state"] == "running":
            raise ValueError("A prep run is already going.")
        if _look["state"] == "running":
            raise ValueError("The Look Filter is scanning. Wait for it to finish.")
        _job.update(state="running", log="", folder=folder, started=time.time())

    def work():
        try:
            what = (f"with SAM 3.1, {', '.join(sam['include'])}"
                    + (f" without {', '.join(sam['exclude'])}" if sam["exclude"] else "")
                    + (" (drawn or furry characters)" if sam.get("drawn") else "")) if sam is not None \
                else (charmask.KIND_LABELS_DRAWN[kind] if who == "drawn" else automask.KINDS[kind]).lower() \
                + {"photo": "", "drawn": ", with the character finder",
                   "auto": ", photo or drawn decided for each picture"}[who]
            _log(f"Making masks: {what}"
                 + (" - only your subject where there are several people" if subject_only else "")
                 + (" (replacing masks already there)" if overwrite else " (masks already there are kept)") + "\n\n")
            subject = det = None
            if subject_only:
                emb = face_embedder()
                vecs = [v for v in (emb.embed_pil(_open_rgb(b))[0] for b in bases) if v is not None]
                if not vecs:
                    raise ValueError("no face found in the Look Filter's baselines")
                subject, det = ((lambda crop: emb.embed_pil(crop)[0]), vecs), _face_detector()
            if sam is not None:
                from fizgig.dataset import sammask
                c = sammask.make_folder(folder, sam["include"], sam["exclude"], sam["path"], overwrite=overwrite,
                                        subject=subject, detector=det, log=_log, drawn=bool(sam.get("drawn")))
            elif who == "photo":
                c = automask.make_folder(folder, kind, MODELS_DIR, overwrite=overwrite, subject=subject, detector=det,
                                         log=_log)
            else:
                chars = charmask.CharacterFinder(charmask.find_model(MODELS_DIR))
                heads = _heads_for(kind)
                try:
                    c = automask.make_folder(folder, kind, MODELS_DIR, overwrite=overwrite, subject=subject,
                                             detector=det, log=_log, who=who, characters=chars, heads=heads)
                finally:
                    chars.close()
            _log(f"\nDone: {c['made']} mask(s) written"
                 + (f", {c['kept']} already there kept" if c["kept"] else "")
                 + (f", {c['no_person']} picture(s) with nothing to mask" if c["no_person"] else "")
                 + (f", {c['skipped']} skipped" if c["skipped"] else "")
                 + ". Check them on the Captions tab (Show mask); paint over any that are wrong.\n"
                 + ("To remake the ones kept, tick Replace masks already there and press Make masks again.\n"
                    if c["kept"] else ""))
            end = "done"
        except Exception as e:
            _log(f"\nERROR: making masks failed — {type(e).__name__}: {e}\n")
            end = "failed"
        with _lock:
            _job["state"] = end
    threading.Thread(target=work, daemon=True).start()
    return {"ok": True}


def quick_preview(folder, kind, who, shown=()):
    """The quick finder's preview: 2 random pictures not shown yet with their masks over them, nothing written
    (sammask.preview_pictures picks them). Returns ([(name, PIL picture, share masked, "photo" / "character", a note:
    what was done or why nothing was found)], the next round's names or None)."""
    from fizgig.dataset import automask, charmask, sammask
    from fizgig.user_data import MODELS_DIR
    import numpy as np
    from PIL import Image
    why = quick_check(kind, who)
    if why:
        raise ValueError(why)
    pics, fresh = sammask.preview_pictures(folder, shown)
    seg = automask.Segmenter(automask.ensure_model(MODELS_DIR, lambda m: None)) if who != "drawn" else None
    chars = charmask.CharacterFinder(charmask.find_model(MODELS_DIR)) if who != "photo" else None
    heads = _heads_for(kind) if who != "photo" else None
    out = []
    for p in pics:
        with Image.open(p) as im:
            img = upright(im).convert("RGB")
        cls = seg.classes(img) if seg is not None else None
        char = chars.mask(img) if chars is not None else None
        drawn = who == "drawn" or (who == "auto" and not charmask.looks_like_photo(cls, char))
        note = ""
        if not drawn:
            m = automask.mask_from(cls, kind)
            if float(m.mean()) < automask.MIN_COVER:
                note = "no person found" if float(automask.mask_from(cls, "person").mean()) < 0.01 \
                    else "nothing of that kind found"
        else:
            m, note, _ = charmask.drawn_mask(char, kind, lambda: [f.bbox for f in (heads.detect_from_pil(img) or [])])
            if m is None:
                m = np.zeros(char.shape, np.float32)
        out.append((*sammask.overlay(os.path.basename(p), img, m), "character" if drawn else "photo", note))
    return out, fresh


def _open_rgb(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.convert("RGB")
