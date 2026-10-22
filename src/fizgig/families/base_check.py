"""Whether a family's Base DiT file is the model's own released Base (FamilyDescription.standard_base).

A family that previews on a separate checkpoint (MOSS-Image on SFT) does so because its own Base is too soft to judge
a LoRA on. Any other Base - a fine-tune such as Kromoss, a Fizgig fine-tune, SFT itself - is the model the LoRA will be
used on, so previews belong on it. One tensor's sum identifies the released Base (a norm weight, which quantised files
keep as it was); a Fizgig fine-tune keeps its norms frozen, so its save's fizgig_finetune metadata names it.
"""
import functools
import os


@functools.lru_cache(maxsize=32)
def _check(path, mtime, size, key, want):
    from safetensors import safe_open
    with safe_open(path, "pt") as f:
        if (f.metadata() or {}).get("fizgig_finetune"):
            return False
        hit = next((k for k in f.keys() if k == key or k.endswith("." + key)), None)
        if hit is None:
            return None                       # a layout without that tensor: unknown, treated as the Base
        return abs(float(f.get_tensor(hit).double().sum()) - want) < 1e-6


def matches(path, fingerprint):
    """True = the file carries this (tensor name, sum) fingerprint, False = it doesn't, None = unknown."""
    if not fingerprint or not path or not os.path.isfile(path):
        return None
    try:
        st = os.stat(path)
        r = _check(os.path.abspath(path), st.st_mtime, st.st_size, *fingerprint)
        return r
    except Exception:
        return None


def checkpoint_takes_speed(desc, path):
    """Whether previews on this preview checkpoint can take the family's speed LoRA (checkpoint_speed_except): any
    checkpoint but the one with that fingerprint (MOSS-Image: Kromoss yes, SFT no). Unset = none do."""
    if not desc.checkpoint_speed_except or not path:
        return False
    return matches(path, desc.checkpoint_speed_except) is False


def is_standard_base(desc, path):
    """True = the released Base, False = another model, None = unknown (no fingerprint, no file, unreadable)."""
    if not desc.standard_base or not path or not os.path.isfile(path):
        return None
    try:
        st = os.stat(path)
        return _check(os.path.abspath(path), st.st_mtime, st.st_size, *desc.standard_base)
    except Exception:
        return None
