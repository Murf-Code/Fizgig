"""Image Prep's "Trim tags that don't fit" taggers, by the id the page saves: "wd-eva02" (wd_tagger.py, the default:
anime and general art, Danbooru-style tags) and "jtp3" (jtp3_tagger.py: furry art, e621 tags). Both modules offer the
same things: LABEL, SIZE_GB, KEEP_SCORE (the keep score a fresh pick starts at), find_model / download / info, and a
Tagger whose scores(img) and categories feed wd_tagger.trim."""
from fizgig.dataset import jtp3_tagger, wd_tagger

DEFAULT = "wd-eva02"
TAGGERS = {"wd-eva02": wd_tagger, "jtp3": jtp3_tagger}
NAMES = {"wd-eva02": "WD EVA02 Large v3 (general anime)", "jtp3": "JTP-3 Hydra (furry / e621)"}


def pick(name):
    """A saved tagger id, checked: an unknown or missing one is the default."""
    name = str(name or "").strip()
    return name if name in TAGGERS else DEFAULT


def get(name):
    return TAGGERS[pick(name)]


def info(models_dir):
    """{id: {path, fetch, gb, label, name, keep_score}} for the page."""
    return {k: {**m.info(models_dir), "name": NAMES[k], "keep_score": m.KEEP_SCORE} for k, m in TAGGERS.items()}
