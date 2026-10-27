"""The Captions tab's tag captioners: the two taggers Image Prep's "Trim tags that don't fit" already uses
(taggers.py: SmilingWolf's WD EVA02 Large v3, wd_tagger.py, and RedRocket's JTP-3 Hydra, jtp3_tagger.py) write a
whole caption of tags instead of trimming one. The model files are the same ones in the same folder, so a tagger
downloaded for Trim is ready here too (and the other way round).

A caption is the tags scored at least the min score, the surest first, comma-separated, at most max tags of them.
Scores are each tagger's own: WD's probability (TagGUI shows the same number) and JTP-3's confidence (2p - 1, as its
demo shows it; jtp3_tagger.Tagger.confidence). For JTP-3 a tag scores at least as high as a tag that implies it, as
Hydra's own captioner does by default ("inherit"), so "canine" comes with "fox". Character tags can have a lower
score of their own; rating tags (WD: general / sensitive / questionable / explicit, JTP-3: safe / questionable /
explicit) are left out unless asked for, and then only the likeliest one is written. Tags are written with spaces
(as TagGUI and Hydra do) or with underscores; brackets are written as they are (Fizgig captions don't escape them).
The trigger word is put in front by the caption worker, as for every captioner; a tag that is the trigger word is
not written twice."""
from fizgig.dataset import jtp3_tagger, taggers
from fizgig.dataset.wd_tagger import norm

# the Captions tab's Model dropdown: name -> taggers.py id
WD_MODEL = "WD EVA02 Large v3 tagger (anime tags)"
JTP3_MODEL = "JTP-3 Hydra tagger (furry / e621 tags)"
MODELS = {WD_MODEL: "wd-eva02", JTP3_MODEL: "jtp3"}
MIN_SCORE = {"wd-eva02": 0.60, "jtp3": 0.30}      # WD: Chris's TagGUI setting; JTP-3: its demo's line
MAX_TAGS = 255
STYLES = ("spaces", "underscores")
CHARACTER = 4                                     # both taggers' character category (Danbooru's and e621's)
WD_RATING = 9
JTP3_RATINGS = {"safe", "questionable", "explicit"}


def tagger_id(model):
    """The taggers.py id of a Captions tab model name, else "" (Florence-2, Qwen3-VL)."""
    return MODELS.get(str(model or ""), "")


def settings(raw, tid):
    """The page's tag settings, checked: {min_score, max_tags, style, ratings, char_score (None: as min_score)}.
    ValueError with what to fix."""
    raw = raw if isinstance(raw, dict) else {}

    def score(key, default, what):
        v = raw.get(key)
        if v is None or str(v).strip() == "":
            return default
        try:
            f = float(v)
        except (TypeError, ValueError):
            f = -1.0
        if not 0.0 < f < 1.0:
            raise ValueError(f"{what}: a number between 0 and 1, like {MIN_SCORE[tid]:.2f}.")
        return f
    try:
        n = int(float(raw.get("max_tags") if str(raw.get("max_tags", "")).strip() else MAX_TAGS))
    except (TypeError, ValueError):
        n = 0
    if n < 1:
        raise ValueError("Max tags: a whole number, 1 or more.")
    style = str(raw.get("style") or "spaces")
    return {"min_score": score("min_score", MIN_SCORE[tid], "Min score"), "max_tags": n,
            "style": style if style in STYLES else "spaces", "ratings": bool(raw.get("ratings")),
            "char_score": score("char_score", None, "Character tags from")}


def scored_tags(tagger, img):
    """[(tag as the model names it, category, score)] for one picture: every tag the tagger knows."""
    if isinstance(tagger, jtp3_tagger.Tagger):
        raw = tagger.raw(img)
        s = jtp3_tagger.inherit({norm(t): tagger.confidence(v) for t, v in raw.items()}, tagger.implies)
        return [(t, tagger.meta.get(t, (0, []))[0], s.get(norm(t), 0.0)) for t in tagger.labels]
    s = tagger.scores(img)
    return [(n, c, s.get(norm(n), 0.0)) for n, c in tagger.tags]


def is_rating(tid, tag, category):
    return category == WD_RATING if tid == "wd-eva02" else tag in JTP3_RATINGS


def write_tag(tag, style):
    """A model tag as the caption writes it: spaces for underscores (kaomoji like ^_^ kept), or the model's own."""
    if style == "underscores":
        return tag.replace(" ", "_")
    return tag.replace("_", " ") if len(tag) > 3 else tag


def caption(tags, tid, cfg, trigger=""):
    """The caption (without the trigger word) from scored_tags' list and checked settings."""
    chosen, rating = [], None
    for tag, cat, sc in tags:
        if is_rating(tid, tag, cat):
            if rating is None or sc > rating[2]:
                rating = (tag, cat, sc)
            continue
        floor = cfg["char_score"] if cat == CHARACTER and cfg["char_score"] is not None else cfg["min_score"]
        if sc >= floor:
            chosen.append((tag, cat, sc))
    if cfg["ratings"] and rating is not None:
        chosen.append(rating)
    chosen.sort(key=lambda t: -t[2])
    trig = norm(trigger) if trigger and trigger.strip() else ""
    out, seen = [], set()
    for tag, _c, _s in chosen:
        n = norm(tag)
        if n == trig or n in seen:
            continue
        seen.add(n)
        out.append(write_tag(tag, cfg["style"]))
        if len(out) >= cfg["max_tags"]:
            break
    return ", ".join(out)


def load(tid, models_dir, log=print):
    """The downloaded tagger, loaded (GPU when there is one, else CPU). ValueError when it isn't downloaded."""
    mod = taggers.get(tid)
    folder = mod.find_model(models_dir)
    if not folder:
        raise ValueError(f"The {mod.LABEL} isn't downloaded yet: press Download under the Model box first.")
    return mod.Tagger(folder, log=log)


def info(models_dir):
    """{model name: {id, path, fetch, gb, label, min_score}} for the Captions tab."""
    out = {}
    for name, tid in MODELS.items():
        out[name] = {**taggers.get(tid).info(models_dir), "id": tid, "min_score": MIN_SCORE[tid]}
    return out
