"""Image Prep's crops keep their picture's caption: a close-up cut from photo.png gets photo.txt's caption as its own
(FaceCrop_001.txt, Crop_paw_001.txt), and a crop that replaces its picture keeps it. Optionally ("Trim tags that
don't fit"), a tag caption loses the tags the chosen tagger no longer sees on the crop (taggers.py: the WD tagger,
wd_tagger.py, or JTP-3 Hydra for furry / e621 captions, jtp3_tagger.py; the rules are wd_tagger.trim's): tags are only
ever removed, never added; the trigger word, the first tag, rating / meta tags and tags the tagger doesn't know
(artists, your own) always stay; character / species tags stay only when it sees them on the crop. Sentence captions are copied as written (tag_variants.split_tags
decides which is which)."""
import os

from fizgig.dataset import wd_tagger

CAPTION_EXTS = (".txt", ".caption")         # looked for after the dataset's own caption extension


def find_caption(image_path, ext=".txt"):
    """The picture's caption file: the dataset's caption extension first, then .txt / .caption. None without one."""
    stem = os.path.splitext(image_path)[0]
    for e in dict.fromkeys((ext or ".txt", *CAPTION_EXTS)):
        p = stem + e
        if os.path.isfile(p):
            return p
    return None


def read(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def write(path, text):
    tmp = path + ".fizgig-tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


CLOSE_UP = "close-up"                                  # e621's (and Danbooru's) tag
_CLOSE_UP_NAMES = {"close-up", "close up", "closeup"}


class CropCaptions:
    """Image Prep's caption settings: copy (a crop gets its picture's caption), trim with a taggers.py Tagger (None:
    no trimming), keep_score (a general tag stays at or above it), ext (the dataset's caption extension), keep (the
    trigger word), closeup (a crop's tag caption gets "close-up"), add_score (with a tagger: tags it is at least
    this sure of on the crop are added, wd_tagger.missing; None: none are)."""

    def __init__(self, copy=True, ext=".txt", keep=(), tagger=None, keep_score=wd_tagger.KEEP_SCORE, overwrite=False,
                 closeup=False, add_score=None):
        self.copy, self.overwrite = bool(copy), bool(overwrite)
        self.ext = ext if str(ext or "").startswith(".") else "." + str(ext or "txt")
        self.keep = [k for k in keep if k]
        self.tagger, self.keep_score = tagger, float(keep_score)
        self.closeup = bool(closeup)
        self.add_score = None if add_score is None else float(add_score)

    def text_for(self, caption, crop_img):
        """(caption, dropped): what the crop's caption says."""
        return self._judge(caption, crop_img)[:2]

    def _judge(self, caption, crop_img):
        """(caption, dropped, the tagger's scores on the crop or None)."""
        from fizgig.dataset.tag_variants import split_tags
        if self.tagger is None or crop_img is None or split_tags(caption) is None:
            return caption, [], None
        t = self.tagger
        scores = t.scores(crop_img)
        text, dropped = wd_tagger.trim(caption, scores, t.categories, self.keep, self.keep_score,
                                       who=getattr(t, "who", wd_tagger.WHO), stand_ins=getattr(t, "stand_ins", None))
        return text, dropped, scores

    def caption_for(self, caption, crop_img, cropped=True):
        """(caption, dropped, added): the crop's caption - trimmed (text_for), the tags the tagger is sure of added
        (add_score), and for a crop (cropped) "close-up" added to a tag caption without it."""
        from fizgig.dataset.tag_variants import split_tags
        text, dropped, scores = self._judge(caption, crop_img)
        tags = split_tags(text)
        added = []
        if tags is not None and scores is not None and self.add_score is not None:
            t = self.tagger
            under = any("_" in x for x in tags) and not any(" " in x for x in tags)    # long_hair style
            added = [n.replace(" ", "_") if under else n for n in wd_tagger.missing(
                tags, scores, t.categories, self.add_score, who=getattr(t, "who", wd_tagger.WHO),
                names=getattr(t, "names", None), aliases=getattr(t, "aliases", None))]
        if tags is not None and cropped and self.closeup and \
                not any(wd_tagger.norm(x) in _CLOSE_UP_NAMES for x in tags + added):
            added.append(CLOSE_UP)
        if added:
            text = ", ".join(tags + added)
        return text, dropped, added

    @staticmethod
    def _say(log, out, dropped, added):
        if dropped:
            log(f"  caption: dropped {', '.join(dropped)}\n")
        if added:
            log(f"  caption: added {', '.join(added)}\n")

    def to_crop(self, source_image, crop_image, log, crop_img=None):
        """A new crop (crop_image, its pixels crop_img) beside its picture: the picture's caption, written as the
        crop's. A caption already there for the crop is left alone. Returns the caption path written, or None."""
        if not self.copy:
            return None
        src = find_caption(source_image, self.ext)
        if not src:
            return None
        out = os.path.splitext(crop_image)[0] + self.ext
        if os.path.exists(out) and not self.overwrite:
            log(f"  {os.path.basename(out)} already there, kept\n")
            return None
        text, dropped, added = self.caption_for(read(src), crop_img)
        write(out, text)
        self._say(log, out, dropped, added)
        return out

    def in_place(self, source_image, output_image, originals_dir, log, crop_img=None, cropped=None):
        """A crop that replaced its picture (Face close-ups only): the caption follows a new name, and when it was
        cropped (cropped; default: crop_img given) it gets close-up and, with trimming on (crop_img), loses the tags
        that don't fit (its first version going to originals/ with the picture, unless originals_dir is None -
        replacing). A picture only resized keeps its caption as written."""
        if not self.copy:
            return None
        src = find_caption(source_image, self.ext)
        if not src:
            return None
        out = os.path.splitext(output_image)[0] + os.path.splitext(src)[1]
        cropped = crop_img is not None if cropped is None else bool(cropped)
        text, dropped, added = self.caption_for(read(src), crop_img if cropped else None, cropped)
        if os.path.normcase(os.path.abspath(out)) != os.path.normcase(os.path.abspath(src)):
            if os.path.exists(out) and not self.overwrite:
                log(f"  {os.path.basename(out)} already there, kept\n")
                return None
            write(out, text)
        elif dropped or added:
            if originals_dir:
                import shutil
                os.makedirs(originals_dir, exist_ok=True)
                shutil.copy2(src, os.path.join(originals_dir, os.path.basename(src)))
            write(out, text)
        self._say(log, out, dropped, added)
        return out
