"""Timestep focus (fizgig.training.timestep_spread): Default draws exactly what each family drew before (the same
values and the same random numbers used, checked against the e2a98c0a formulas written out here), every preset moves
the timestep distribution the way its name says (means, medians and the clean-end share), a pinned window (the
validation loss's fixed noise levels) never moves, a training step with a preset runs on CPU (SDXL and Anima, batch 1
and batched), and the setting goes Training tab -> preset -> command -> trainer with its checks."""
import math
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fizgig.families import form, launch  # noqa: E402
from fizgig.families.registry import FAMILIES, get  # noqa: E402
from fizgig.training import timestep_spread as TS  # noqa: E402

SHAPES = {"krea2": (1, 16, 8, 8), "klein": (1, 128, 8, 8), "qwen_image21": (1, 16, 8, 8), "flux3": (1, 128, 8, 8),
          "zimage": (1, 16, 8, 8), "anima": (1, 16, 8, 8), "sdxl": (1, 4, 8, 8)}
OFFERED = set(SHAPES)


# ---- the samplers as they were at e2a98c0a (before Timestep focus), written out -----------------------------------
def _shift(t, s):
    return (t * s) / (1.0 + (s - 1.0) * t)


def _old_t(key, gen, n_tokens, min_t, max_t):
    if key == "krea2":
        m = (1.15 - 0.5) / (6400 - 256)
        t = _shift(torch.randn(1, generator=gen).sigmoid(), float(np.exp(m * n_tokens + (0.5 - m * 256))))
        if min_t > 0.0 or max_t < 1.0:
            lo, hi = max(0.0, float(min_t)), min(1.0, float(max_t))
            t = lo + t * max(hi - lo, 1e-6)
        return float(t.reshape(-1)[0])
    if key == "klein":
        m = (1.15 - 0.5) / (4096 - 256)
        t = _shift(torch.randn(1, generator=gen).sigmoid(), float(np.exp(m * n_tokens + (0.5 - m * 256))))
        return float((t * (float(max_t) - float(min_t)) + float(min_t)).reshape(-1)[0])
    if key == "qwen_image21":
        from fizgig.qwen_image21 import sampling as S
        t = torch.sigmoid(torch.randn(1, generator=gen)).item()
        mu = S.calculate_mu(n_tokens)
        return min_t + (max_t - min_t) * (math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0)))
    if key == "flux3":
        from fizgig.flux3 import sampling as S
        return min_t + (max_t - min_t) * S.shift(torch.sigmoid(torch.randn(1, generator=gen)).item())
    if key == "zimage":
        s = _shift(torch.sigmoid(torch.randn(1, generator=gen)), 3.0)
        return float((min_t + (max_t - min_t) * s).reshape(-1)[0])
    if key == "anima":
        return float((min_t + (max_t - min_t) * torch.sigmoid(torch.randn(1, generator=gen))).reshape(-1)[0])
    if key == "sdxl":
        lo = int(round(min_t * 999))
        hi = max(lo + 1, int(round(max_t * 999)) + 1)
        return int(torch.randint(lo, min(hi, 1000), (1,), generator=gen).item())
    raise KeyError(key)


NOISE_FIRST = {"krea2", "klein", "qwen_image21", "flux3"}         # noise_latents draws the noise, then t


def _tokens(key, shape):
    h, w = shape[-2:]
    return (h // 2) * (w // 2) if key == "krea2" else h * w


def _t_of(st, key):
    t = st["t"]
    t = float(t.reshape(-1)[0]) if torch.is_tensor(t) else float(t)
    return t


def _draws(key, preset, n=3000, min_t=0.0, max_t=1.0, seed=0):
    drv = get(key).load_driver()
    drv.set_options({} if preset is None else {"timestep_spread": preset})
    g = torch.Generator().manual_seed(seed)
    out = [_t_of(drv.noise_latents(torch.zeros(SHAPES[key]), g, min_t=min_t, max_t=max_t), key) for _ in range(n)]
    return np.array(out) / (999.0 if key == "sdxl" else 1.0), g


# ---- which families ---------------------------------------------------------------------------------------------
def test_which_families_offer_it():
    assert {k for k, d in FAMILIES.items() if d.timestep_spread} == OFFERED | {"zimage_base"}   # Base shares Z-Image's
    for k in ("moss_image", "minimax", "acestep15xl"):       # their own Clean-end share / schedules
        assert FAMILIES[k].timestep_spread == ()
    assert get("sdxl").timestep_spread == ("default", "detail", "composition")      # its even draw is balanced
    for k in OFFERED - {"sdxl"}:
        assert get(k).timestep_spread == TS.PRESETS


# ---- Default: bit for bit what it was -------------------------------------------------------------------------------
@pytest.mark.parametrize("key", sorted(OFFERED))
@pytest.mark.parametrize("opts", [None, "default", "Default"])
def test_default_is_the_old_sampling(key, opts):
    drv = get(key).load_driver()
    drv.set_options({} if opts is None else {"timestep_spread": opts})
    g_new, g_old = torch.Generator().manual_seed(11), torch.Generator().manual_seed(11)
    shape = SHAPES[key]
    for i in range(200):
        min_t, max_t = (0.15 if i % 3 == 0 else 0.0), (0.8 if i % 4 == 0 else 1.0)
        lat_new, lat_old = torch.randn(shape, generator=g_new), torch.randn(shape, generator=g_old)
        assert torch.equal(lat_new, lat_old)
        st = drv.noise_latents(lat_new, g_new, min_t=min_t, max_t=max_t)
        if key in NOISE_FIRST:
            torch.randn(shape, generator=g_old)
        want = _old_t(key, g_old, _tokens(key, shape), min_t, max_t)
        if key not in NOISE_FIRST:
            torch.randn(shape, generator=g_old)                        # the noise the driver drew after t
        assert _t_of(st, key) == want, (i, _t_of(st, key), want)
    assert torch.equal(g_new.get_state(), g_old.get_state())         # the same random numbers used, so a resume's
                                                                       # draws (saved generator state) are unchanged


def test_default_helpers_are_identity():
    z = torch.randn(1000, generator=torch.Generator().manual_seed(3))
    assert torch.equal(TS.logit_normal(z, "default"), z.sigmoid())
    assert torch.equal(TS.logit_normal(z, "detail", 0.4, 0.4), z.sigmoid())          # pinned window: never moved
    u = torch.rand(1000, generator=torch.Generator().manual_seed(4))
    assert TS.even(u, "default") is u and TS.even(u, "balanced") is u


# ---- each preset moves the curve its way ----------------------------------------------------------------------------
@pytest.mark.parametrize("key", sorted(OFFERED))
def test_presets_move_the_distribution(key):
    base, _ = _draws(key, "default")
    clean = lambda a: float((a < 0.5).mean())                          # share of steps at the clean end
    det, _ = _draws(key, "detail")
    comp, _ = _draws(key, "composition")
    assert det.mean() < base.mean() - 0.05 and np.median(det) < np.median(base) - 0.05
    assert clean(det) > clean(base) + 0.08
    assert comp.mean() > base.mean() + 0.05 and np.median(comp) > np.median(base) + 0.05
    assert clean(comp) < clean(base) or clean(base) < 0.05
    assert np.quantile(det, 0.25) < np.quantile(base, 0.25) and np.quantile(comp, 0.75) > np.quantile(base, 0.75)
    for a in (det, comp):
        assert a.min() >= 0.0 and a.max() <= 1.0
    if "balanced" in get(key).timestep_spread:
        bal, _ = _draws(key, "balanced")
        assert abs(np.median(bal) - np.median(base)) < 0.04            # centred where the model usually trains
        assert bal.std() > base.std() * 1.15                           # ...spread wider
        tails = lambda a: float(((a < 0.15) | (a > 0.9)).mean())
        assert tails(bal) > tails(base)


@pytest.mark.parametrize("key", sorted(OFFERED))
def test_window_still_holds_and_pinned_never_moves(key):
    for p in get(key).timestep_spread:
        a, _ = _draws(key, p, n=400, min_t=0.2, max_t=0.6)
        assert a.min() >= 0.2 - 1e-3 and a.max() <= 0.6 + 1e-3
    # the validation loss pins the window to one noise level: every preset gives the same t from the same draws
    ref, gref = _draws(key, "default", n=50, min_t=0.375, max_t=0.375)
    for p in get(key).timestep_spread:
        got, g = _draws(key, p, n=50, min_t=0.375, max_t=0.375)
        assert np.array_equal(got, ref) and torch.equal(g.get_state(), gref.get_state())


@pytest.mark.parametrize("key", sorted(OFFERED))
def test_presets_repeat_from_the_same_seed(key):
    """A resume / Save & stop restores the generator: the same seed gives the same draws with a preset too."""
    for p in get(key).timestep_spread:
        a, ga = _draws(key, p, n=100, seed=5)
        b, gb = _draws(key, p, n=100, seed=5)
        assert np.array_equal(a, b) and torch.equal(ga.get_state(), gb.get_state())


def test_flow_presets_are_the_shift_and_width():
    """For the flow families a preset is the family's own shift times SHIFT[p], the logit's spread times WIDTH[p]."""
    from fizgig.zimage.driver import ZImageDriver
    for p in TS.PRESETS:
        g1, g2 = torch.Generator().manual_seed(9), torch.Generator().manual_seed(9)
        got = ZImageDriver._sample_sigma(500, g1, 0.0, 1.0, p)
        z = torch.randn(500, generator=g2) * TS.WIDTH[p]
        want = _shift(z.sigmoid(), 3.0 * TS.SHIFT[p])
        assert torch.allclose(got, want, atol=1e-5)


# ---- a CPU training step with a preset ------------------------------------------------------------------------------
@pytest.mark.parametrize("preset", ["detail", "composition"])
def test_sdxl_trains_with_a_preset(tmp_path, preset):
    from checks.test_batch_size import _UNet, _check_family
    drv = get("sdxl").load_driver()
    drv.set_options({"timestep_spread": preset})
    g = torch.Generator().manual_seed(5)
    items = [{"hidden_states": torch.randn(77 * c, 2048, generator=g), "pooled": torch.randn(1280, generator=g)}
             for c in (1, 2, 1, 2)]
    torch.manual_seed(0)
    _check_family(drv, _UNet(), tmp_path, items, (4, 8, 8), 2, 1e-5)       # batched = mean of singles, a step taken
    unet = _UNet()
    opt = torch.optim.SGD(unet.parameters(), lr=1e-2)
    gen = torch.Generator().manual_seed(1)
    ts = []
    for it in items:
        cond = {"hidden_states": it["hidden_states"][None], "pooled": it["pooled"][None]}
        loss, info = drv.training_loss(unet, torch.randn(1, 4, 8, 8, generator=gen), cond, gen)
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
        opt.zero_grad()
        ts.append(info["t"])
    assert all(0.0 <= t <= 1.0 for t in ts)


@pytest.mark.parametrize("preset", ["composition", "balanced"])
def test_anima_trains_with_a_preset(tmp_path, preset):
    pytest.importorskip("einops")
    from checks.test_batch_size import _check_family
    from fizgig.anima.driver import DIT_CONFIG
    from fizgig.anima.model import Anima
    torch.manual_seed(0)
    dit = Anima(**dict(DIT_CONFIG, model_channels=64, num_blocks=2, num_heads=4, adaln_lora_dim=16)).to(torch.bfloat16)
    for p in dit.parameters():
        p.requires_grad_(p.dim() == 2 and p.shape[0] == 64)
    g = torch.Generator().manual_seed(3)
    items = [{"qwen": torch.randn(nq, 1024, generator=g), "t5_ids": torch.randint(2, 500, (nt,), generator=g)}
             for nq, nt in ((7, 9), (3, 4), (5, 2), (2, 6))]
    drv = get("anima").load_driver()
    drv.set_options({"timestep_spread": preset})
    _check_family(drv, dit, tmp_path, items, (16, 8, 8), 2, 5e-2)       # batched = mean of singles, a step taken


class _ZDiT(torch.nn.Module):
    """Stands in for Z-Image's DiT: per image, a channel mix of the latent plus the caption's mean and the time."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(16, 16) * 0.1)
        self.c = torch.nn.Linear(2560, 16)

    def forward(self, xs, t, caps, cap_weights=None):
        return [torch.einsum("oc,cfhw->ofhw", self.w, x.float()) + self.c(c.float().mean(0))[:, None, None, None]
                + float(ti) for x, c, ti in zip(xs, caps, t)]


@pytest.mark.parametrize("preset", ["detail", "balanced", "composition"])
def test_zimage_trains_with_a_preset(preset):
    drv = get("zimage").load_driver()
    drv.set_options({"timestep_spread": preset})
    torch.manual_seed(0)
    dit = _ZDiT()
    opt = torch.optim.SGD(dit.parameters(), lr=1e-2)
    gen = torch.Generator().manual_seed(2)
    before = [p.detach().clone() for p in dit.parameters()]
    for _ in range(3):
        loss, info = drv.training_loss(dit, torch.randn(1, 16, 8, 8, generator=gen),
                                       {"cap": torch.randn(1, 6, 2560, generator=gen)}, gen)
        assert torch.isfinite(loss) and 0.0 <= info["t"] <= 1.0
        loss.backward()
        opt.step()
        opt.zero_grad()
    assert any(not torch.equal(a, p) for a, p in zip(before, dit.parameters()))


# ---- Training tab, presets, command, trainer ------------------------------------------------------------------------
def _rows(desc):
    return {r["key"]: r for s in form.form(desc)["sections"] for r in s["rows"] if r}


def test_training_tab_row():
    for k, d in FAMILIES.items():
        rows = _rows(d)
        if not d.timestep_spread:
            assert "TIMESTEP_SPREAD" not in rows, k
            continue
        r = rows["TIMESTEP_SPREAD"]
        assert r["kind"] == "choice" and r["default"] == "Model default" and r["label"] == "Timestep focus"
        assert r["choices"] == [TS.LABELS[p] for p in d.timestep_spread]
    assert form.DEFAULTS["TIMESTEP_SPREAD"] == "Model default"
    krea = get("krea2")
    for p in TS.PRESETS:
        assert form.state(krea, {**form.DEFAULTS, "TIMESTEP_SPREAD": TS.LABELS[p]})["notes"]["TIMESTEP_SPREAD"] == \
            TS.HINTS[p]
    assert "Pick one of" in form.state(get("sdxl"), {**form.DEFAULTS, "TIMESTEP_SPREAD": "Balanced"})["notes"][
        "TIMESTEP_SPREAD"]
    assert "TIMESTEP_SPREAD" not in form.state(get("moss_image"), dict(form.DEFAULTS))["notes"]


def test_presets_keep_it():
    krea, sdxl = get("krea2"), get("sdxl")
    got = form.apply_preset(krea, dict(form.DEFAULTS), {"TIMESTEP_SPREAD": "Detail"})
    assert got["TIMESTEP_SPREAD"] == "Detail (clean end)"
    got = form.apply_preset(krea, dict(form.DEFAULTS), {"TIMESTEP_SPREAD": "composition"})      # any case
    assert got["TIMESTEP_SPREAD"] == "Composition (noisy end)"
    got = form.apply_preset(sdxl, {**form.DEFAULTS, "TIMESTEP_SPREAD": "Detail (clean end)"}, {"TIMESTEP_SPREAD": "Balanced"})
    assert got["TIMESTEP_SPREAD"] == "Detail (clean end)"               # a preset from another model: not offered here, kept


def test_command_flag_and_start_checks():
    for k in sorted(OFFERED):
        d = get(k)
        assert launch.spread_flags(d, {"TIMESTEP_SPREAD": "Default"}) == []          # Default: the command as before
        assert launch.spread_flags(d, {}) == []
        assert launch.spread_flags(d, {"TIMESTEP_SPREAD": "Detail"}) == ["--timestep_spread", "detail"]
        assert launch.spread_problems(d, {"TIMESTEP_SPREAD": "Composition"}) == []
        assert launch.spread_problems(d, {"TIMESTEP_SPREAD": "Sideways"})
    assert launch.spread_problems(get("sdxl"), {"TIMESTEP_SPREAD": "Balanced"})
    assert launch.spread_flags(get("sdxl"), {"TIMESTEP_SPREAD": "Balanced"}) == []
    moss = get("moss_image")
    assert launch.spread_flags(moss, {"TIMESTEP_SPREAD": "Detail"}) == [] and \
        launch.spread_problems(moss, {"TIMESTEP_SPREAD": "Detail"}) == []           # not its setting: ignored


def test_full_command(tmp_path):
    krea = get("krea2")
    img = tmp_path / "imgs"
    img.mkdir()
    (img / "a.png").write_bytes(b"x")
    (img / "a.txt").write_text("zwxem a photo", encoding="utf-8")
    models = {}
    for mf in krea.model_files:
        if mf.required or mf.pref_key == krea.training_adapter:
            f = tmp_path / f"{mf.pref_key}.safetensors"
            f.write_bytes(b"0")
            models[mf.pref_key] = str(f)

    def plan(v):
        inp = form.inputs(krea, {**form.DEFAULTS, "LORA_NAME": "t", "LORA_OUTPUT_DIR": str(tmp_path / "out"),
                                 "DATASET_CONFIG": str(tmp_path / "dataset.toml"), **v},
                          models=models, python=sys.executable, repo_dir=str(tmp_path), image_folder=str(img),
                          cache_root=str(tmp_path / "cache"), samples_dir=str(tmp_path / "out" / "sample"))
        return launch.plan(krea, inp)
    p = plan({"TIMESTEP_SPREAD": "Composition"})
    cmd = p.stages[-1].cmd
    assert cmd[cmd.index("--timestep_spread") + 1] == "composition"
    assert not any("Timestep focus" in x for x in p.problems)
    assert "--timestep_spread" not in plan({}).stages[-1].cmd                  # Default: no flag
    assert any("Timestep focus" in x for x in plan({"TIMESTEP_SPREAD": "Upside down"}).problems)


def test_trainer_flag_and_checks():
    from fizgig.families import train
    base = ["--family", "krea2", "--dit", "x", "--dataset_config", "d", "--output_dir", "o", "--output_name", "n"]
    assert train.setup_parser().parse_args(base).timestep_spread == "default"
    assert train.setup_parser().parse_args(base + ["--timestep_spread", "Detail"]).timestep_spread == "detail"
    with pytest.raises(SystemExit):
        train.setup_parser().parse_args(base + ["--timestep_spread", "sideways"])
    src = open(train.__file__, encoding="utf-8").read()
    assert '{"timestep_spread": a.timestep_spread} if a.timestep_spread != "default"' in src   # Default adds nothing
    for fam, val, ok in (("krea2", "detail", True), ("sdxl", "composition", True), ("sdxl", "balanced", False),
                         ("moss_image", "detail", False), ("moss_image", "default", True), ("krea2", "nope", False)):
        d = get(fam)
        drv = d.load_driver()
        drv.set_options({"timestep_spread": val})
        if ok:
            train._check_timestep_spread(d, drv)
            assert drv.timestep_spread == val
        else:
            with pytest.raises(RuntimeError):
                train._check_timestep_spread(d, drv)


def test_labels_old_and_new_names_map_to_same_values():
    for k in TS.PRESETS:
        assert TS.key(TS.LABELS[k]) == k and TS.key(k.capitalize()) == k
    assert TS.key("Model default") == "default" and TS.LABELS["detail"] == "Detail (clean end)"
