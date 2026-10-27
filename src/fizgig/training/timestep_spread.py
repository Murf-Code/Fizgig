"""Timestep focus: a few named presets that move which noise levels training spends its steps on, for the families
whose own sampler has no such dial (MOSS-Image keeps its Clean-end share, MiniMax H3 its own; ACE-Step has none).

Every preset is built on the sampler parameters the families already have, so it reads as "the same curve, moved":

* flow matching (Krea 2, Klein, Qwen Image, Z-Image, Flux 3, Anima): t = shift(sigmoid(z)), z ~ N(0, 1), where the
  shift (a fixed number or one from the image size; Anima's is 1) is t -> s t / (1 + (s - 1) t). A shift adds log s
  to the logit, so a preset multiplies the family's own shift (Detail 0.4x: towards the clean end; Composition 2.5x:
  towards the noise) or widens the curve around its usual centre (Balanced: z scaled by 1.6, every noise level
  closer to an even share). Drawn exactly as before (one normal draw per timestep), so a run is as repeatable as ever.
* SDXL (DDPM, 1000 discrete timesteps drawn evenly): the even draw u goes through the same shift (Detail 0.4,
  Composition 2.5) before it picks the timestep. Its usual draw is already even, so it has no Balanced.

Default is the family's sampling untouched: the same code path and the same random draws, bit for bit. A pinned
window (min == max, the validation loss's fixed noise levels) never moves, so validation stays comparable.

Written for Fizgig from the general idea (timestep shift / logit-normal sampling); no code from other trainers."""
import math

PRESETS = ("default", "detail", "balanced", "composition")
LABELS = {"default": "Default", "detail": "Detail", "balanced": "Balanced", "composition": "Composition"}
HINTS = {"default": "The model's usual mix of noise levels.",
         "detail": "More steps at the clean end: faces, skin, texture and fine detail.",
         "balanced": "Steps spread more evenly over every noise level.",
         "composition": "More steps at the noisy end: layout, pose and overall shapes."}
FLOW = PRESETS                                  # the flow-matching families offer all four
DDPM = ("default", "detail", "composition")     # SDXL: its even draw is already balanced

SHIFT = {"default": 1.0, "detail": 0.4, "balanced": 1.0, "composition": 2.5}     # x the family's own shift
WIDTH = {"default": 1.0, "detail": 1.0, "balanced": 1.6, "composition": 1.0}     # x the logit-normal's spread


def key(value) -> str:
    """'Detail', 'detail', ' DETAIL ' -> 'detail'; '' / None -> 'default'. ValueError for anything else."""
    v = str(value or "").strip().lower()
    if not v:
        return "default"
    v = v.split(" ")[0]
    if v not in PRESETS:
        raise ValueError(f"unknown timestep focus {value!r} (one of {', '.join(LABELS.values())})")
    return v


def label(value) -> str:
    return LABELS[key(value)]


def active(preset, min_t=0.0, max_t=1.0) -> bool:
    """Whether a preset moves this draw: not Default, and not a window pinned to one noise level."""
    return key(preset) != "default" and float(max_t) > float(min_t)


def logit_normal(z, preset="default", min_t=0.0, max_t=1.0):
    """sigmoid(z) for a standard-normal draw z (a tensor), the preset's move applied first: the flow families'
    logit-normal base, before their own shift and window. Default: z.sigmoid() exactly as the family drew it."""
    if not active(preset, min_t, max_t):
        return z.sigmoid()
    p = key(preset)
    return (z * WIDTH[p] + math.log(SHIFT[p])).sigmoid()


def even(u, preset="default"):
    """An even draw u in [0, 1) moved by the preset's shift (SDXL's timestep draw). Balanced / Default: u as is."""
    s = SHIFT[key(preset)]
    return u if s == 1.0 else s * u / (1.0 + (s - 1.0) * u)
