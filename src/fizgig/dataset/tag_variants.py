"""Tag shuffle, keep-first-N, tag dropout and caption dropout over a frozen text encoder.

The text encoder never trains, so each caption is encoded once and cached. Shuffling or dropping tags every step (the
kohya-style recipe) would need the encoder at every step; instead the cache pass writes a few VARIANTS of each tag
caption - its tags reordered and some dropped, the first N (and the folder's trigger word) always kept, first - and
each step every image picks one of them. Caption dropout is one more cached variant, the empty caption, picked with
its own probability.

Picks are a pure function of (run seed, epoch, image key, which repeat of the image this is in the epoch): no RNG
state to save, so a resumed run (a whole epoch or a mid-epoch Save & stop) picks exactly what the first run would
have, and at Batch Size > 1 each image of a batch picks its own.

Files: beside each item's text cache `<stem>_<arch>_te.safetensors`, `<stem>_<arch>_tags.safetensors` holds the
variants' conditioning (`v<i>__<key>`) and, with caption dropout, the empty caption's (`e__<key>`); its metadata keeps
the caption it was made from, the recipe, and each variant's text. A variant whose source caption no longer matches
the text cache (a caption fix, auto-recaption, an edited .txt) is never used - the image trains on its own caption
until the variants are rebuilt - and a rebuild re-encodes only the variant texts the file doesn't already hold.

Natural-language captions (no commas, or long comma-separated clauses) are left as written: no variants, one console
note. Caption dropout still applies to them.

Written for Fizgig from the idea (kohya-ss sd-scripts' shuffle_caption / keep_tokens / caption_tag_dropout_rate /
caption_dropout_rate); no code taken from another trainer.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)

TAGS_SUFFIX = "_tags.safetensors"
TE_SUFFIX = "_te.safetensors"
MAX_VARIANTS = 32
AVG_WORDS_PROSE = 4.0        # comma-separated parts longer than this on average read as sentences, not tags


@dataclass(frozen=True)
class TagSettings:
    shuffle: bool = False
    keep_first: int = 1
    tag_dropout: float = 0.0          # 0..1, each tag after the kept ones
    caption_dropout: float = 0.0      # 0..1, the whole caption, per image per step
    variants: int = 4
    keep_words: tuple = field(default_factory=tuple)

    @classmethod
    def from_params(cls, shuffle=False, keep_first=1, tag_dropout=0.0, caption_dropout=0.0, variants=4,
                    keep_words=None):
        def frac(x):                   # a fraction, 0..1 (the dataset TOML's percent is divided by the dataset)
            return min(max(float(x or 0), 0.0), 1.0)
        return cls(shuffle=bool(shuffle), keep_first=max(0, int(keep_first if keep_first is not None else 1)),
                   tag_dropout=frac(tag_dropout), caption_dropout=frac(caption_dropout),
                   variants=min(MAX_VARIANTS, max(1, int(variants or 4))),
                   keep_words=tuple(w.strip() for w in (keep_words or ()) if w and w.strip()))

    @property
    def varies(self) -> bool:
        """Tag variants are made (shuffle or tag dropout)."""
        return self.shuffle or self.tag_dropout > 0

    @property
    def active(self) -> bool:
        return self.varies or self.caption_dropout > 0

    def recipe(self) -> str:
        """What a variant's text depends on besides the caption and its index (not the count, not caption dropout):
        a changed count reuses the variants it shares."""
        if not self.varies:
            return ""
        kw = ",".join(sorted(w.lower() for w in self.keep_words))
        return f"s{int(self.shuffle)}|k{self.keep_first}|d{self.tag_dropout:.4f}|w{kw}"

    def describe(self) -> str:
        bits = []
        if self.shuffle:
            bits.append("shuffle tags")
        if self.varies:
            bits.append(f"keep first {self.keep_first}")
        if self.tag_dropout:
            bits.append(f"tag dropout {self.tag_dropout * 100:g}%")
        if self.caption_dropout:
            bits.append(f"caption dropout {self.caption_dropout * 100:g}%")
        if self.varies:
            bits.append(f"{self.variants} variants per caption")
        return ", ".join(bits)


def settings_of(ds) -> Optional[TagSettings]:
    """A dataset's TagSettings (ImageDataset.tag_settings), or None when it has none active."""
    s = getattr(ds, "tag_settings", None)
    return s if s is not None and s.active else None


# ---- captions --------------------------------------------------------------------------------------------------
_SENTENCE = re.compile(r"[.!?]\s+\S")


def split_tags(caption: str) -> Optional[list]:
    """A tag caption's tags in order, or None for natural language (left as written)."""
    parts = [p.strip() for p in (caption or "").split(",")]
    parts = [p for p in parts if p]
    if len(parts) < 2:
        return None
    words = [len(p.split()) for p in parts]
    if sum(words) / len(words) > AVG_WORDS_PROSE or any(_SENTENCE.search(p) for p in parts):
        return None
    return parts


def _rng(*parts) -> random.Random:
    h = hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).digest()
    return random.Random(int.from_bytes(h[:16], "big"))


def make_variant(caption: str, s: TagSettings, index: int, seed: int = 0) -> str:
    """Variant `index` of a caption: the kept head (trigger words first, then the first N tags) as written, the rest
    with each tag dropped at tag_dropout and the survivors shuffled. Natural language comes back unchanged."""
    tags = split_tags(caption)
    if tags is None or not s.varies:
        return caption
    kw = {w.lower() for w in s.keep_words}
    trig = [t for t in tags if t.lower() in kw]
    head = trig + [t for t in tags[:s.keep_first] if t.lower() not in kw]
    tail = [t for t in tags[s.keep_first:] if t.lower() not in kw]
    r = _rng(seed, s.recipe(), caption, index)
    if s.tag_dropout:
        tail = [t for t in tail if r.random() >= s.tag_dropout]
    if s.shuffle:
        r.shuffle(tail)
    return ", ".join(head + tail)


def variant_captions(caption: str, s: TagSettings, seed: int = 0) -> list:
    """The caption's variants (s.variants of them), [] when it gets none (natural language, or no shuffle/dropout)."""
    if not s.varies or split_tags(caption) is None:
        return []
    return [make_variant(caption, s, i, seed) for i in range(s.variants)]


# ---- files -----------------------------------------------------------------------------------------------------
def variants_path(te_path: str) -> str:
    te_path = str(te_path)
    base = te_path[:-len(TE_SUFFIX)] if te_path.endswith(TE_SUFFIX) else os.path.splitext(te_path)[0]
    return base + TAGS_SUFFIX


def _header(path):
    from safetensors import safe_open
    try:
        with safe_open(path, framework="pt") as f:
            return f.metadata() or {}, list(f.keys())
    except Exception:
        return None, []


def te_caption(te_path) -> Optional[str]:
    md, _ = _header(te_path)
    return None if md is None else md.get("caption1")


def write_variants(encode: Callable[[list], list], te_path: str, caption: str, s: TagSettings, *, arch: str = "",
                   seed: int = 0, clean: Callable = None) -> int:
    """Write (or bring up to date) one item's variants file. encode(captions) -> conditioning dicts (CPU tensors).
    Reuses what the file and the item's own text cache already hold; returns how many captions it encoded. A caption
    with nothing to vary and no caption dropout gets no file (an old one is removed)."""
    from safetensors.torch import load_file, save_file
    path = variants_path(te_path)
    wanted = variant_captions(caption, s, seed)
    if not wanted and not s.caption_dropout:
        if os.path.exists(path):
            os.remove(path)
        return 0
    have = {}                                   # caption text -> {key: tensor}
    md, keys = _header(path) if os.path.exists(path) else (None, [])
    if md is not None and md.get("fizgig_tags") == "1":
        old = load_file(path)
        try:
            texts = json.loads(md.get("captions") or "[]")
        except ValueError:
            texts = []
        for i, t in enumerate(texts):
            d = {k[len(f"v{i}__"):]: v for k, v in old.items() if k.startswith(f"v{i}__")}
            if d:
                have.setdefault(t, d)
        e = {k[3:]: v for k, v in old.items() if k.startswith("e__")}
        if e:
            have.setdefault("", e)
    if caption in wanted and caption not in have and os.path.exists(te_path):
        own = load_file(te_path)                # the original order came up: its own encoding
        have[caption] = {k[len("cond__"):]: v for k, v in own.items() if k.startswith("cond__")}
    need = [t for t in dict.fromkeys(wanted + ([""] if s.caption_dropout else [])) if t not in have]
    if need:
        for t, c in zip(need, encode(need)):
            have[t] = {k: (clean(v, k) if clean else v.detach().cpu().contiguous()) for k, v in c.items()}
    sd, used = {}, set()

    def put(name, v):                 # two variants with the same text: their own copies (safetensors refuses shared)
        v = v.contiguous()
        if id(v) in used or v.data_ptr() in used:
            v = v.clone()
        used.update((id(v), v.data_ptr()))
        sd[name] = v
    for i, t in enumerate(wanted):
        for k, v in have[t].items():
            put(f"v{i}__{k}", v)
    if s.caption_dropout:
        for k, v in have[""].items():
            put(f"e__{k}", v)
    meta = {"fizgig_tags": "1", "architecture": arch, "source": caption, "recipe": s.recipe(),
            "captions": json.dumps(wanted, ensure_ascii=False), "empty": "1" if s.caption_dropout else "0"}
    tmp = path + ".tmp"
    save_file(sd, tmp, metadata=meta)
    os.replace(tmp, path)
    return len(need)


def build_for_datasets(datasets, encode: Callable[[list], list], *, arch: str = "", num_workers: int = 1,
                       batch: int = 8, clean: Callable = None) -> dict:
    """The cache pass: every item's variants for datasets with tag settings, the encoder already loaded. Encodes in
    groups of `batch` captions. -> {"items", "encoded", "prose"}; logs one line per dataset (and one for prose)."""
    out = {"items": 0, "encoded": 0, "prose": 0}
    for i, ds in enumerate(datasets):
        s = settings_of(ds)
        if s is None:
            continue
        items = [it for b in ds.retrieve_text_encoder_output_cache_batches(num_workers) for it in b]
        pending = []                  # encode across items in groups: gather what each needs first

        def flush():
            if not pending:
                return
            texts = list(dict.fromkeys(t for _, _, need in pending for t in need))
            got = {}
            for j in range(0, len(texts), max(1, batch)):
                chunk = texts[j:j + batch]
                for t, c in zip(chunk, encode(chunk)):
                    got[t] = c
            for te, cap, _need in pending:
                out["encoded"] += write_variants(
                    lambda ts: [got[t] if t in got else encode([t])[0] for t in ts], te, cap, s, arch=arch,
                    clean=clean)
            pending.clear()

        prose = 0
        for it in items:
            te = it.text_encoder_output_cache_path
            if not s.varies or split_tags(it.caption) is None:
                prose += bool(s.varies and it.caption.strip())
            pending.append((te, it.caption, _needed(te, it.caption, s)))
            if sum(len(n) for _, _, n in pending) >= batch * 4:
                flush()
        flush()
        out["items"] += len(items)
        out["prose"] += prose
        _remove_orphans(ds)
        logger.info(f"[tags] dataset [{i}]: {s.describe()} - variants ready for {len(items)} caption(s)")
        if prose:
            logger.info(f"[tags] {prose} caption(s) read as sentences, not comma-separated tags: they keep their "
                        "words and order (caption dropout still applies to them)")
    if out["items"]:
        logger.info(f"[tags] encoded {out['encoded']} new caption variant(s); the rest were already cached")
    return out


def _needed(te_path, caption, s: TagSettings) -> list:
    """The variant texts write_variants would encode for this item (none it can reuse)."""
    wanted = variant_captions(caption, s) + ([""] if s.caption_dropout else [])
    if not wanted:
        return []
    have = set()
    path = variants_path(te_path)
    md, _ = _header(path) if os.path.exists(path) else (None, [])
    if md is not None and md.get("fizgig_tags") == "1":
        try:
            have |= set(json.loads(md.get("captions") or "[]"))
        except ValueError:
            pass
        if md.get("empty") == "1":
            have.add("")
    if os.path.exists(te_path):
        have.add(caption)
    return [t for t in dict.fromkeys(wanted) if t not in have]


def _remove_orphans(ds):
    """Variants files whose item's text cache is gone (an image removed from the folder)."""
    import glob
    d = getattr(ds, "cache_directory", None)
    arch = getattr(ds, "architecture", "")
    if not d or not os.path.isdir(d):
        return
    for p in glob.glob(os.path.join(glob.escape(d), f"*_{arch}{TAGS_SUFFIX}")):
        if not os.path.exists(p[:-len(TAGS_SUFFIX)] + TE_SUFFIX):
            try:
                os.remove(p)
            except OSError:
                pass


# ---- training: the per-step pick ---------------------------------------------------------------------------------
ORIG, EMPTY = "orig", "empty"


def _unit(seed, epoch, key, occ):
    h = hashlib.sha256(f"tags\x1f{seed}\x1f{epoch}\x1f{key}\x1f{occ}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64, int.from_bytes(h[8:16], "big") / 2 ** 64


def _stamp(path):
    """A file's identity for the picker's header cache: a rewrite changes it even within one clock tick (variants
    files are replaced, so a new inode)."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size, st.st_ino


class VariantPicker:
    """One dataset's per-step choice: the original caption, one of its cached variants, or the empty caption."""

    def __init__(self, settings: TagSettings, seed: int = 0):
        self.s = settings
        self.seed = int(seed or 0)
        self._info = {}               # variants path -> (te mtime, var mtime, (usable variants, empty usable))
        self.stale = set()            # items whose variants don't match their caption (said once)

    def info(self, te_path):
        path = variants_path(te_path)
        vm, tm = _stamp(path), _stamp(te_path)
        if vm is None:
            return 0, False
        hit = self._info.get(path)
        if hit and hit[0] == tm and hit[1] == vm:
            return hit[2]
        md, _ = _header(path)
        n, empty = 0, False
        if md is not None and md.get("fizgig_tags") == "1":
            empty = md.get("empty") == "1"
            if self.s.varies and md.get("recipe") == self.s.recipe() and md.get("source") == te_caption(te_path):
                try:
                    n = min(len(json.loads(md.get("captions") or "[]")), self.s.variants)
                except ValueError:
                    n = 0
            elif self.s.varies:
                self.stale.add(path)
        self._info[path] = (tm, vm, (n, empty))
        return n, empty

    def choose(self, key, te_path, epoch, occ=0):
        u1, u2 = _unit(self.seed, epoch, key, occ)
        n, empty = self.info(te_path)
        if self.s.caption_dropout and u1 < self.s.caption_dropout and empty:
            return EMPTY
        if self.s.varies and n:
            return min(n - 1, int(u2 * n))
        return ORIG

    @staticmethod
    def load(te_path, choice) -> dict:
        """The chosen variant's tensors under the text cache's own key names (cond__...)."""
        from safetensors import safe_open
        pre = "e__" if choice == EMPTY else f"v{int(choice)}__"
        out = {}
        with safe_open(variants_path(te_path), framework="pt") as f:
            for k in f.keys():
                if k.startswith(pre):
                    out["cond__" + k[len(pre):]] = f.get_tensor(k)
        return out


def summary(datasets) -> list:
    """Console lines for a training run: what each dataset does and how many of its images have usable variants."""
    lines = []
    for i, ds in enumerate(datasets):
        bm = getattr(ds, "batch_manager", None)
        vp = getattr(bm, "variants", None)
        if vp is None:
            continue
        items = {it.text_encoder_output_cache_path: it for b in bm.buckets.values() for it in b}
        ok = empty = 0
        prose = 0
        for te in items:
            n, e = vp.info(te)
            ok += n > 0
            empty += e
            if vp.s.varies and not n:
                cap = te_caption(te)
                prose += cap is not None and split_tags(cap) is None
        lines.append(f"[tags] dataset [{i}]: {vp.s.describe()} (seed {vp.seed})")
        if vp.s.varies:
            missing = len(items) - ok - prose
            lines.append(f"[tags] {ok} of {len(items)} image(s) train with tag variants"
                         + (f"; {prose} with sentence captions keep theirs as written" if prose else "")
                         + (f"; {missing} have none cached for their current caption and train on it as written "
                            "(start a new run with Prepare the cache ticked to make them)" if missing > 0 else ""))
        if vp.s.caption_dropout and empty < len(items):
            lines.append(f"[tags] caption dropout: {len(items) - empty} image(s) have no cached empty caption and "
                         "always keep theirs (start a new run with Prepare the cache ticked)")
    return lines
