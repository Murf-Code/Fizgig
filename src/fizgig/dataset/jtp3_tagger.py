"""Project RedRocket's JTP-3 Hydra (Apache-2.0, https://huggingface.co/RedRocket/Hydra, formerly RedRocket/JTP-3), an
e621 tagger for furry art: Image Prep's "Trim tags that don't fit" can ask it, instead of the WD tagger, which of a
crop's copied tags it still sees. Downloaded once, on demand, into models/jtp-3-hydra (the 1.0 GB weights, the tag list
with e621 categories and implications, and the e621 rename list); run with plain PyTorch through the network modules
vendored in jtp3_hydra/ (credit and licence there) on the GPU when there is one, else the CPU; freed when prep ends.

The same trim rules as the WD tagger (wd_tagger.trim): only general tags (e621 category 0) the tagger knows and scores
under the keep score are dropped. Species (5), character (4), artist, copyright (3), meta (7, which holds the rating
tags safe / questionable / explicit), lore (8), tags it doesn't know and your own always stay; tags are never added.

e621 names: a caption's "long_hair", "long hair" and "Long Hair" all match (wd_tagger.norm). A tag e621 renamed since
your captions were written (e.g. "animal pussy", now "animal_vulva") matches through RedRocket's rename list. Like
Hydra's own "inherit" mode, a tag scores at least as high as any tag that implies it, so "breasts" stays while the
tagger sees "big breasts", and "hair" while it sees "long hair"."""
import csv
import os
import threading
import time

from fizgig.dataset.wd_tagger import norm
from fizgig.utils import hf_public

REPO = "RedRocket/Hydra"
REVISION = "cfa9b0a1ffcf2b8df8553be7673210fd60fba23b"    # pinned: the weights, tag list and vendored code agree
WEIGHTS = "models/jtp-3-hydra.safetensors"
TAGS = "data/jtp-3-hydra-tags.csv"
ALIASES = "data/aliases-2024-04-07.txt"
FILES = (WEIGHTS, TAGS, ALIASES)
FOLDER = "jtp-3-hydra"
LABEL = "JTP-3 Hydra tagger"
SIZE_GB = 1.0
GENERAL = 0
KEEP_SCORE = 0.20                        # a general tag stays if the tagger scores it at least this on the crop
CATEGORIES = ("general", "artist", "contributor", "copyright", "character", "species", "invalid", "meta", "lore")
PATCH = 16
MAX_PATCHES = 1024                       # NaFlex: up to 1024 16-pixel patches, about 512x512, aspect kept

_lock = threading.Lock()
_fetch = {"state": "", "error": "", "at": 0.0}


def model_dir(models_dir):
    return os.path.join(models_dir, FOLDER)


def find_model(models_dir):
    """The tagger's folder when all its files are there, else ""."""
    d = model_dir(models_dir)
    return d if all(os.path.isfile(os.path.join(d, f)) for f in FILES) else ""


def download(models_dir):
    """Fetch the files into models/jtp-3-hydra on a worker thread; fetch_state() follows it."""
    with _lock:
        if _fetch["state"] == "running":
            return
        _fetch.update(state="running", error="", at=time.time())

    def work():
        try:
            for f in FILES[::-1]:                     # the small lists first
                hf_public.download(repo_id=REPO, filename=f, revision=REVISION, local_dir=model_dir(models_dir))
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
            gb = max((os.path.getsize(os.path.join(r, n)) for r, _s, fs in os.walk(d) for n in fs
                      if n.endswith(".incomplete")), default=0) / 2 ** 30
        except OSError:
            pass
    return {"state": st["state"], "error": st["error"], "gb": round(gb, 2), "of": SIZE_GB}


def info(models_dir):
    return {"path": find_model(models_dir), "fetch": fetch_state(models_dir), "gb": SIZE_GB, "label": LABEL}


def read_tags(path):
    """jtp-3-hydra-tags.csv -> {e621 tag: (category number, [tags it implies])}."""
    with open(path, encoding="utf-8", newline="") as f:
        return {row["tag"]: (int(row["category"]), row["implications"].split()) for row in csv.DictReader(f)}


def read_aliases(path):
    """The rename list ('model_tag older_name' per line) -> {normalised older name: normalised model tag}."""
    out = {}
    if not path or not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                out.setdefault(norm(parts[1]), norm(parts[0]))
    return out


def inherit(scores, implies):
    """Each tag's score raised to the highest score of the tags that imply it (directly or through a chain), as
    Hydra's "inherit" mode does: a caption's "canine" isn't trimmed while the tagger sees "fox"."""
    out = dict(scores)
    done = {}

    def best(t, seen=()):
        if t in done:
            return done[t]
        v = out.get(t, 0.0)
        for child in implied_by.get(t, ()):
            if child not in seen:
                v = max(v, best(child, seen + (t,)))
        done[t] = v
        return v
    implied_by = {}
    for t, parents in implies.items():
        for p in parents:
            implied_by.setdefault(p, []).append(t)
    for t in list(out):
        out[t] = best(t)
    return out


def fit_patches(w, h, patch=PATCH, max_patches=MAX_PATCHES):
    """The (width, height) the picture is resized to: whole 16-pixel patches, at most max_patches of them, the
    aspect kept, never enlarged."""
    pw, ph = max(1, w // patch), max(1, h // patch)
    if pw * ph <= max_patches:
        return pw * patch, ph * patch
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if max(1, int(w * mid) // patch) * max(1, int(h * mid) // patch) <= max_patches:
            lo = mid
        else:
            hi = mid
    return max(1, int(w * lo) // patch) * patch, max(1, int(h * lo) // patch) * patch


def preprocess(img, background=(0, 0, 0)):
    """A PIL picture as JTP-3 takes it: sRGB, transparency on black (its training setting), resized (Lanczos) to
    whole 16-pixel patches, cut into a (1, rows, cols, 768) grid of patches scaled to -1..1."""
    import io
    import numpy as np
    import torch
    from PIL import Image
    icc = img.info.get("icc_profile")
    if icc:
        try:
            from PIL import ImageCms
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            mode = "RGBA" if "A" in img.getbands() else "RGB"
            img = ImageCms.profileToProfile(img.convert(mode), src, ImageCms.createProfile("sRGB"), outputMode=mode)
        except Exception:
            pass                              # a broken profile: read the pixels as they are
    if "A" in img.getbands() or img.mode == "P":
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, tuple(background) + (255,))
        img = Image.alpha_composite(bg, img)
    img = img.convert("RGB")
    size = fit_patches(*img.size)
    if size != img.size:
        img = img.resize(size, Image.LANCZOS, reducing_gap=3.0)
    a = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy())
    rows, cols = size[1] // PATCH, size[0] // PATCH
    a = a.reshape(rows, PATCH, cols, PATCH, 3).permute(0, 2, 1, 3, 4).reshape(1, rows, cols, PATCH * PATCH * 3)
    return a.float().div_(127.5).sub_(1.0)


ARCH = "naflexvit_so400m_patch16_siglip+rr_hydra"


def make_net(n_labels, device="cpu", dtype=None):
    """JTP-3's network, empty: SigLIP2 So400m NaFlex (27 blocks of 1152) and the Hydra pool (2048 wide, 64 per head,
    6144 feed-forward) with one SwiGLU output per tag, as RedRocket's load_model builds "rr_hydra"."""
    import torch
    from fizgig.dataset.jtp3_hydra.siglip2 import NaFlexVit
    from fizgig.dataset.jtp3_hydra.pool import HydraPool
    from fizgig.dataset.jtp3_hydra.head import SwiGLUHead, ExtremumPool
    dtype = dtype or torch.bfloat16
    net = NaFlexVit(device=device, dtype=dtype)
    net.attn_pool = HydraPool(n_labels, 2048, 64, input_dim=1152, mid_blocks=0, ff_dim=6144, device=device,
                              dtype=dtype)
    net.head = SwiGLUHead(n_labels, 2048, logit=False, device=device, dtype=dtype)
    net.emb_head = ExtremumPool()
    return net


def convert(sd):
    """The checkpoint's weights under the network's names (its older names for the pool's feed-forward and the
    output heads); a "validation" table, if any, left out."""
    sd = {k: v for k, v in sd.items() if k != "validation"}
    for a, b in (("ff_in.weight", "ff.proj_in.weight"), ("ff_out.weight", "ff.proj_out.weight"),
                 ("ff_norm.weight", "ff.norm.weight"), ("ff_norm.bias", "ff.norm.bias")):
        sd["attn_pool." + b] = sd.pop("attn_pool." + a)
    sd["head.proj.weight"] = sd.pop("attn_pool.out_proj.weight").mT
    return sd


def build(path, device="cpu", dtype=None):
    """The network with the weights loaded, its tag names in output order, and the background colour for
    transparency."""
    import torch
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        if meta.get("modelspec.architecture") != ARCH:
            raise RuntimeError(f"{os.path.basename(path)} isn't the JTP-3 Hydra checkpoint.")
        sd = {k: f.get_tensor(k) for k in f.keys()}
    labels = meta["classifier.labels"].split()
    net = make_net(len(labels), "cpu", torch.bfloat16)
    net.load_state_dict(convert(sd), strict=True)
    del sd
    net.eval().requires_grad_(False)
    # its own precision (bf16): about 2.5 GB of RAM at the peak on the CPU against 3 GB in float32, as quick
    net.to(device=device, dtype=dtype or torch.bfloat16)
    background = {"white": 255, "grey": 127}.get(meta.get("classifier.background", "black"), 0)
    return net, labels, (background,) * 3


class Tagger:
    """The loaded tagger: scores(img) -> {normalised tag: probability}; categories: {normalised tag: e621 category}."""

    def __init__(self, folder, log=print, net=None, labels=None, device=None):
        self.meta = read_tags(os.path.join(folder, TAGS))
        self.aliases = read_aliases(os.path.join(folder, ALIASES))
        self.background = (0, 0, 0)
        if net is None:
            import torch
            device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            log(f"Loading {LABEL} ({'GPU' if device.startswith('cuda') else 'CPU'}) ...\n")
            net, labels, self.background = build(os.path.join(folder, WEIGHTS), device)
        self.net, self.labels = net, list(labels)
        self.implies = {norm(t): [norm(p) for p in imp] for t, (_c, imp) in self.meta.items() if imp}
        self.categories = {norm(t): self.meta.get(t, (GENERAL, []))[0] for t in self.labels}
        for old, new in self.aliases.items():
            if new in self.categories and old not in self.categories:
                self.categories[old] = self.categories[new]

    def raw(self, img):
        """{e621 tag: probability} straight from the network."""
        import torch
        w = self.net.embeds.proj.weight            # (pos_embed may be widened to float32 on the CPU)
        x = preprocess(img, self.background).to(device=w.device, dtype=w.dtype)
        with torch.inference_mode():
            probs = self.net(x)[0].float().cpu().tolist()
        return dict(zip(self.labels, probs))

    @staticmethod
    def confidence(p):
        """The network's probability as RedRocket's JTP-3 demo shows it: 2p - 1, floored at 0. Its unsure middle
        (p = 0.5) is 0 and a sure tag 1; the demo lists tags from 0.30 up."""
        return max(0.0, 2.0 * float(p) - 1.0)

    def scores(self, img):
        """{normalised tag: confidence 0-1}, implied tags raised to their best implier, older e621 names included."""
        s = inherit({norm(t): self.confidence(v) for t, v in self.raw(img).items()}, self.implies)
        for old, new in self.aliases.items():
            if new in s and old not in s:
                s[old] = s[new]
        return s

    def close(self):
        self.net = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
