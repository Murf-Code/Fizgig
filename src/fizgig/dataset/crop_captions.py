"""Image Prep's crops keep their picture's caption: a close-up cut from photo.png gets photo.txt's caption as its own
(FaceCrop_001.txt, Crop_paw_001.txt), and a crop that replaces its picture keeps it. Optionally ("Trim tags that
don't fit"), a tag caption loses the general tags the chosen tagger no longer sees on the crop (taggers.py: the WD
tagger, wd_tagger.py, or JTP-3 Hydra for furry / e621 captions, jtp3_tagger.py): tags are only ever removed, never added; the trigger word, the first tag, character / rating / meta tags and tags the tagger
doesn't know (species, artists, your own) always stay. Sentence captions are copied as written (tag_variants.split_tags
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


class CropCaptions:
    """Image Prep's caption settings: copy (a crop gets its picture's caption), trim with a taggers.py Tagger (None:
    no trimming), keep_score (a general tag stays at or above it), ext (the dataset's caption extension), keep (the
    trigger word)."""

    def __init__(self, copy=True, ext=".txt", keep=(), tagger=None, keep_score=wd_tagger.KEEP_SCORE, overwrite=False):
        self.copy, self.overwrite = bool(copy), bool(overwrite)
        self.ext = ext if str(ext or "").startswith(".") else "." + str(ext or "txt")
        self.keep = [k for k in keep if k]
        self.tagger, self.keep_score = tagger, float(keep_score)

    def text_for(self, caption, crop_img):
        """(caption, dropped): what the crop's caption says."""
        from fizgig.dataset.tag_variants import split_tags
        if self.tagger is None or crop_img is None or split_tags(caption) is None:
            return caption, []
        return wd_tagger.trim(caption, self.tagger.scores(crop_img), self.tagger.categories, self.keep,
                              self.keep_score)

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
        text, dropped = self.text_for(read(src), crop_img)
        write(out, text)
        if dropped:
            log(f"  caption: dropped {', '.join(dropped)}\n")
        return out

    def in_place(self, source_image, output_image, originals_dir, log, crop_img=None):
        """A crop that replaced its picture (Face close-ups only): the caption follows a new name, and when it was
        cropped (crop_img) and trimming is on, it loses the tags that don't fit (its first version going to originals/
        with the picture, unless originals_dir is None - replacing)."""
        if not self.copy:
            return None
        src = find_caption(source_image, self.ext)
        if not src:
            return None
        out = os.path.splitext(output_image)[0] + os.path.splitext(src)[1]
        text, dropped = self.text_for(read(src), crop_img)
        if os.path.normcase(os.path.abspath(out)) != os.path.normcase(os.path.abspath(src)):
            if os.path.exists(out) and not self.overwrite:
                log(f"  {os.path.basename(out)} already there, kept\n")
                return None
            write(out, text)
        elif dropped:
            if originals_dir:
                import shutil
                os.makedirs(originals_dir, exist_ok=True)
                shutil.copy2(src, os.path.join(originals_dir, os.path.basename(src)))
            write(out, text)
        if dropped:
            log(f"  caption: dropped {', '.join(dropped)}\n")
        return out
