"""Reclaim VRAM: values and optimizer state bit-for-bit the same after a reclaim, the optimizer still steps the same,
the swapper's ring / pending transfers handled, the auto trigger's rules, the launch flags and the UI request."""
import copy
import os
import sys
import types

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fizgig.training import vram_reclaim as vr  # noqa: E402
from fizgig.training.ema import EMAWeights  # noqa: E402
from fizgig.training.step_log import take_request  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "src", "fizgig")


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(16, 16)
        self.base.requires_grad_(False)
        self.register_buffer("scale", torch.linspace(0.5, 1.5, 16))
        self.down = torch.nn.Parameter(torch.randn(4, 16) * 0.1)
        self.up = torch.nn.Parameter(torch.randn(16, 4) * 0.1)

    def forward(self, x):
        return self.base(x) * self.scale + x @ self.down.t() @ self.up.t()


def _opt(kind, params):
    if kind == "adamw":
        return torch.optim.AdamW(params, lr=1e-3)
    if kind == "prodigy":
        from fizgig.training.prodigy import Prodigy
        return Prodigy(params, lr=1.0)
    if kind == "muon":
        from fizgig.training.muon import Muon
        return Muon(params, lr=1e-3)
    raise ValueError(kind)


def _step(model, opt, ema, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(8, 16, generator=g)
    loss = model(x).pow(2).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if ema is not None:
        ema.update()


def _snap(model, opt, ema):
    out = [t.detach().clone() for t in model.state_dict().values()]
    out += [p.grad.clone() for p in model.parameters() if p.grad is not None]
    for st in opt.state.values():
        out += [v.clone() for v in st.values() if torch.is_tensor(v)]
    out += [s.clone() for s in ema.shadow]
    return out


def _same(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)


@pytest.mark.parametrize("kind", ["adamw", "prodigy", "muon"])
def test_reclaim_keeps_values_and_training_bitwise(kind):
    torch.manual_seed(0)
    a = Toy()
    b = copy.deepcopy(a)
    oa, ob = _opt(kind, [a.down, a.up]), _opt(kind, [b.down, b.up])
    ea, eb = EMAWeights(a, 0.9), EMAWeights(b, 0.9)
    for s in range(3):
        _step(a, oa, ea, s)
        _step(b, ob, eb, s)
    params_before = list(a.parameters())
    keys_before = list(oa.state.keys())
    ptr_before = a.down.data_ptr()
    snap = _snap(a, oa, ea)
    res = vr.reclaim(modules=(a,), optimizers=(oa,), extra=(ea.shadow,), device="cpu", probe_windows=False)
    assert res["tensors"] > 0
    _same(snap, _snap(a, oa, ea))                       # bit-for-bit the same values
    assert all(p is q for p, q in zip(params_before, a.parameters()))    # same Parameter objects
    assert all(k is p for k, p in zip(oa.state.keys(), keys_before))   # optimizer state still keyed by them
    assert ea.params[0] is a.down                       # EMA still tracks the live parameters
    assert a.down.data_ptr() != ptr_before              # a fresh allocation
    for s in range(3, 6):                               # and it trains on exactly as the untouched copy
        _step(a, oa, ea, s)
        _step(b, ob, eb, s)
    _same(_snap(a, oa, ea), _snap(b, ob, eb))


def test_views_skipped_and_aliases_kept():
    flat = torch.arange(32, dtype=torch.float32)
    m = torch.nn.Module()
    m.p = torch.nn.Parameter(flat[:16].view(4, 4), requires_grad=False)    # a view into something bigger
    m.w = torch.nn.Parameter(torch.ones(3))
    alias = m.w.data                                    # another tensor object on the same storage
    opt = types.SimpleNamespace(state={"x": {"buf": alias}}, param_groups=[])
    p_ptr = m.p.data_ptr()
    res = vr.reclaim(modules=(m,), optimizers=(opt,), device="cpu", probe_windows=False)
    assert m.p.data_ptr() == p_ptr and res["skipped"] >= 1         # the view stays where it was
    assert alias.data_ptr() == m.w.data_ptr()                      # the alias moved with it, still shared
    alias.add_(1)
    assert torch.equal(m.w.data, torch.full((3,), 2.0))


def test_ring_offloader_rebound():
    flat = torch.arange(64, dtype=torch.uint8)
    w1 = torch.nn.Parameter(flat[0:16].view(torch.float32).view(2, 2), requires_grad=False)
    w2 = torch.nn.Parameter(flat[32:48].view(4, 4), requires_grad=False)
    m = torch.nn.Module()
    m.lin = torch.nn.Linear(2, 2, bias=False)
    m.lin.weight = w1
    m.offloader = types.SimpleNamespace(ring_flat=[flat], ring_param=[[w1, w2]],
                                        copier=types.SimpleNamespace(sync=lambda: setattr(m, "synced", True)))
    v1, v2 = w1.detach().clone(), w2.detach().clone()
    vr.reclaim(modules=(m,), device="cpu", probe_windows=False)
    new = m.offloader.ring_flat[0]
    assert new is not flat and m.synced
    assert torch.equal(w1.data, v1) and torch.equal(w2.data, v2)
    assert w1.data_ptr() == new.data_ptr() and w2.data_ptr() == new.data_ptr() + 32
    assert m.lin.weight is w1


def test_pending_block_transfers_settled_first():
    waited = []

    class Off:
        futures = {3: None, 5: None}

        def _wait_blocks_move(self, k):
            waited.append(k)
            self.futures.pop(k)

    m = torch.nn.Module()
    m.offloader = Off()
    vr.reclaim(modules=(m,), device="cpu", probe_windows=False)
    assert sorted(waited) == [3, 5]


# ---- the automatic trigger ----

def _w(**kw):
    kw.setdefault("warmup_steps", 50)
    return vr.SlowdownWatch(**kw)


def _feed(w, dt, n):
    evs = []
    for _ in range(n):
        evs.append(w.observe(dt))
    return [e for e in evs if e]


def test_warmup_ignored_and_median_adapts():
    w = _w()
    _feed(w, 20.0, 20)                  # a slow first epoch start
    _feed(w, 7.0, 80)
    assert w.baseline == 7.0 and not w.slow
    assert _feed(w, 7.2, 30) == []


def test_slowdown_needs_a_sustained_run():
    w = _w()
    _feed(w, 7.0, 60)
    _feed(w, 13.0, 9)                    # a short slow patch (a preview's aftermath, big buckets)
    assert _feed(w, 7.0, 11) == [] and not w.slow
    evs = _feed(w, 13.0, 20)
    assert evs == ["slow"] and w.slow


def test_one_quick_step_does_not_reset_the_count():
    w = _w()
    _feed(w, 7.0, 60)
    evs = []
    for i in range(30):                  # slow, with a small-bucket step now and then (real runs look like this)
        evs += _feed(w, 9.0 if i % 6 == 5 else 30.0, 1)
    assert evs == ["slow"]


def test_gradual_ramp_is_caught():
    """Chris's run: 7 s creeping up to 35 s over ~150 steps. A moving median would follow it up; the run's best
    speed doesn't."""
    w = _w()
    _feed(w, 7.0, 60)
    evs, fired_at = [], None
    for i in range(150):
        dt = 7.0 + 28.0 * i / 149
        ev = w.observe(dt)
        if ev:
            evs.append(ev)
            fired_at = fired_at if fired_at is not None else dt
    assert evs == ["slow"] and w.slow
    assert w.baseline == pytest.approx(7.0, rel=0.1)
    assert fired_at < 2.0 * 7.0          # caught while it was still only about twice as slow


def test_not_slow_with_free_vram_does_nothing():
    w = _w()
    _feed(w, 7.0, 100)
    assert w.check(free=8 * vr.GB, total=12 * vr.GB) is None


@pytest.mark.parametrize("card_gb", [8, 12, 12.28, 16, 24])
def test_free_threshold_is_flat_0_1_gb(card_gb):
    assert vr.RECLAIM_FREE_GB == 0.1
    assert vr.reclaim_headroom(int(card_gb * vr.GB)) == int(0.1 * vr.GB)
    assert vr.reclaim_headroom(None) == int(0.1 * vr.GB)


def test_free_threshold_preference_override():
    assert vr.reclaim_headroom(12 * vr.GB, 0.5) == int(0.5 * vr.GB)
    assert vr.reclaim_headroom(12 * vr.GB, 0) == int(0.1 * vr.GB)       # 0 / junk: the default
    assert vr.reclaim_headroom(12 * vr.GB, "x") == int(0.1 * vr.GB)


def _slow(**kw):
    w = _w(**kw)
    _feed(w, 7.0, 60)
    assert _feed(w, 30.0, 20) == ["slow"]
    return w


def test_full_card_waits_indefinitely_and_says_so_once():
    w = _slow()
    total = int(12.28 * vr.GB)
    acts = []
    for _ in range(2000):                                   # far past any step limit: still waiting, quietly
        _feed(w, 30.0, 1)
        acts.append(w.check(int(0.05 * vr.GB), total))
    assert acts[0] == "wait_log" and set(acts[1:]) == {"wait"}
    assert acts.count("wait_log") == 1 and w.slow and w.tries == 0


def test_reclaims_when_usage_dips_under_threshold():
    w = _slow()
    total = int(12.28 * vr.GB)
    assert w.check(int(0.09 * vr.GB), total) == "wait_log"    # 12.19 GB used: still full
    assert w.check(int(0.099 * vr.GB), total) == "wait"
    assert w.check(int(0.1 * vr.GB), total) == "reclaim"      # 0.1 GB free: it goes


def test_no_mem_info_reclaims_once_slow_confirmed():
    w = _slow()
    assert w.check(None, None) == "reclaim"


def test_wait_note_again_only_for_a_new_slowdown():
    w = _slow(cooldown=0)
    assert w.check(0, 12 * vr.GB) == "wait_log"
    assert w.check(0, 12 * vr.GB) == "wait"
    assert _feed(w, 7.0, 5) == ["recovered"]                  # it sorted itself out
    assert _feed(w, 30.0, 20) == ["slow"]                     # a new slowdown: one new note
    assert w.check(0, 12 * vr.GB) == "wait_log"
    assert w.check(0, 12 * vr.GB) == "wait"


def test_improved_then_cooldown():
    w = _w(cooldown=200)
    _feed(w, 7.0, 60)
    _feed(w, 13.0, 20)
    w.reclaimed()
    assert _feed(w, 7.1, 20) == ["improved"] and w.last_eval == pytest.approx(7.1)
    assert _feed(w, 13.0, 100) == []          # inside the cooldown: no new slow state
    assert not w.slow
    _feed(w, 7.0, 100)
    assert _feed(w, 13.0, 20) == ["slow"]     # after it, it can fire again


def test_no_improvement_stops_auto():
    w = _w(cooldown=0)
    _feed(w, 7.0, 60)
    _feed(w, 13.0, 20)
    w.reclaimed()
    assert _feed(w, 13.0, 20) == ["no_improvement"] and w.stopped
    assert _feed(w, 13.0, 50) == [] and w.check(8 * vr.GB, vr.GB) is None


def test_manual_no_improvement_keeps_auto():
    w = _w(cooldown=0)
    _feed(w, 7.0, 60)
    w.reclaimed(manual=True)
    assert _feed(w, 13.0, 20) == ["no_improvement"] and not w.stopped and w.tries == 0


def test_max_tries():
    w = _w(cooldown=0, max_tries=3)
    _feed(w, 7.0, 60)
    for _ in range(3):
        assert _feed(w, 13.0, 20) == ["slow"]
        w.reclaimed()
        assert _feed(w, 7.0, 20) == ["improved"]
    assert _feed(w, 13.0, 40) == [] and not w.slow


def test_recovers_by_itself():
    w = _w()
    _feed(w, 7.0, 60)
    _feed(w, 13.0, 20)
    assert w.slow
    assert _feed(w, 7.0, 5) == ["recovered"] and not w.slow


# ---- wiring ----

def test_launch_flags_from_prefs():
    from fizgig.families.launch import RECLAIM_PREF_DEFAULTS, reclaim_flags
    assert reclaim_flags({}) == [] and reclaim_flags(None) == []      # on, defaults: nothing to pass
    assert reclaim_flags(dict(RECLAIM_PREF_DEFAULTS)) == []
    assert reclaim_flags({"auto_reclaim_vram": "0"}) == ["--no-auto_reclaim_vram"]
    assert reclaim_flags({"reclaim_slowdown": "2", "reclaim_max_tries": "5", "reclaim_cooldown": "x"}) == \
        ["--reclaim_slowdown", "2", "--reclaim_max_tries", "5"]


def test_prefs_round_trip(tmp_path, monkeypatch):
    from fizgig.families import launch
    from fizgig.web import server
    monkeypatch.setattr(server, "PREFS_FILE", str(tmp_path / "prefs.json"))
    p = server.load_prefs()                                            # nothing saved: the defaults
    assert {k: p[k] for k in launch.RECLAIM_PREF_DEFAULTS} == launch.RECLAIM_PREF_DEFAULTS
    assert set(launch.RECLAIM_PREF_DEFAULTS) <= set(server.SETTING_PREFS)     # Reset to defaults puts them back
    server.update_prefs(launch.clean_reclaim_prefs(
        {"auto_reclaim_vram": True, "reclaim_slowdown": " 1.8 ", "reclaim_slow_steps": "30.0",
         "reclaim_min_free_gb": ""}))
    p = server.load_prefs()
    assert (p["auto_reclaim_vram"], p["reclaim_slowdown"], p["reclaim_slow_steps"], p["reclaim_min_free_gb"]) == \
        ("1", "1.8", "30", "0.1")
    assert launch.reclaim_flags(p) == ["--reclaim_slowdown", "1.8", "--reclaim_slow_steps", "30"]
    server.update_prefs(launch.clean_reclaim_prefs({"auto_reclaim_vram": False}))
    assert launch.reclaim_flags(server.load_prefs()) == ["--no-auto_reclaim_vram"]
    for bad in ("abc", "0", "-2"):
        with pytest.raises(ValueError):
            launch.clean_reclaim_prefs({"reclaim_cooldown": bad})


def test_old_saved_run_with_form_keys_loads_and_prefs_decide():
    from fizgig.families import form, launch, registry
    desc = registry.get("krea2") or next(iter(registry.FAMILIES.values()))
    old = {"AUTO_RECLAIM_VRAM": False, "RECLAIM_SLOWDOWN": "3", "RECLAIM_SLOW_STEPS": "5",
           "RECLAIM_WARMUP_STEPS": "1", "RECLAIM_MIN_FREE_GB": "9", "RECLAIM_COOLDOWN": "1", "RECLAIM_MAX_TRIES": "9"}
    form.state(desc, {**form.DEFAULTS, **old})                         # loads without an error
    form.apply_preset(desc, dict(form.DEFAULTS), old)
    st = form.inputs(desc, old, models={}, python="python", repo_dir=".", image_folder="", cache_root="",
                     samples_dir="", vram_reclaim={"auto_reclaim_vram": "1"})
    assert not any(k in st for k in launch.LEGACY_RECLAIM_KEYS)        # the old keys are dropped...
    assert launch.reclaim_flags(st["vram_reclaim"]) == []              # ...and Preferences decide


def test_trainer_args_default_on():
    src = open(os.path.join(ROOT, "families", "train.py"), encoding="utf-8").read()
    assert '"--auto_reclaim_vram", action=argparse.BooleanOptionalAction, default=True' in src
    for flag in ("--reclaim_slowdown", "--reclaim_slow_steps", "--reclaim_warmup_steps", "--reclaim_min_free_gb",
                 "--reclaim_cooldown", "--reclaim_max_tries"):
        assert flag in src


def test_settings_live_in_preferences_not_the_form():
    from fizgig.families import form, registry
    assert not any(k.startswith(("AUTO_RECLAIM", "RECLAIM_")) for k in form.DEFAULTS)
    for d in registry.FAMILIES.values():
        keys = {r["key"] for s in form.form(d)["sections"] for r in s["rows"] if r}
        assert not any(k.startswith(("AUTO_RECLAIM", "RECLAIM_")) for k in keys)
    js = open(os.path.join(ROOT, "web", "static", "app.js"), encoding="utf-8").read()
    assert "host.append(vramReclaimCard());" in js and "Advanced: VRAM reclaim" in js
    for key in ("reclaim_slowdown", "reclaim_slow_steps", "reclaim_warmup_steps", "reclaim_min_free_gb",
                "reclaim_cooldown", "reclaim_max_tries"):
        assert f'["{key}",' in js
    assert "b.disabled = !on.checked" in js                            # greyed while auto is off


def test_ui_request_round_trip(tmp_path):
    srv = open(os.path.join(ROOT, "web", "server.py"), encoding="utf-8").read()
    assert 'what not in ("sample", "save", "reclaim")' in srv and '".reclaim_now"' in srv
    html = open(os.path.join(ROOT, "web", "static", "index.html"), encoding="utf-8").read()
    js = open(os.path.join(ROOT, "web", "static", "app.js"), encoding="utf-8").read()
    assert 'id="btnReclaimNow"' in html and 'now("reclaim"' in js
    open(os.path.join(tmp_path, ".reclaim_now"), "w").close()        # what the server drops
    assert take_request(str(tmp_path), "reclaim") is True             # the trainer takes it once
    assert take_request(str(tmp_path), "reclaim") is False


# ---- console notes: one line each ----

def test_say_prints_once(capsys):
    import logging
    from fizgig.training.console_notes import say
    root = logging.getLogger()
    saved = root.handlers[:], root.level
    lg = logging.getLogger("fizgig_test_say_once")
    try:
        root.handlers = [logging.StreamHandler(sys.stderr)]
        root.handlers[0].setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        root.setLevel(logging.INFO)
        say(lg, "[vram] hello there.")
        say(lg, "[vram] careful now.", logging.WARNING)
        out = capsys.readouterr()
        lines = [l for l in (out.out + out.err).splitlines() if "[vram]" in l]
        assert lines == ["INFO:fizgig_test_say_once:[vram] hello there.",
                         "WARNING:fizgig_test_say_once:[vram] careful now."]
        assert lg.handlers == [] and len(root.handlers) == 1      # handlers put back as they were
    finally:
        root.handlers, _ = saved[0], root.setLevel(saved[1])


def test_trainer_vram_messages():
    src = open(os.path.join(ROOT, "families", "train.py"), encoding="utf-8").read()
    assert "vram_reclaim.vram_free_total(device)" in src and "free_and_need" not in src
    assert "Close other GPU apps and it'll recover on its own" not in src
    assert '"[vram] Training spilled into system RAM and slowed down. It\'ll move back "' in src
    assert "Try raising Blocks Swap next run." in src
    for line in src.splitlines():                                  # every [vram] note goes out one way: say()
        if '"[vram]' in line or "f\"[vram]" in line:
            assert "say(logger" in line or "logger.debug" in line, line


def test_collect_takes_a_holder_without_buffers():
    """The families pass FamilyLoRA, which has parameters() but no buffers() (Krea 2 LoKr run, step 14025)."""
    import torch
    from fizgig.training import vram_reclaim

    lin = torch.nn.Linear(4, 4)

    class Holder:                      # like FamilyLoRA: parameters() only
        def parameters(self):
            return list(lin.parameters())

    groups, skipped = vram_reclaim.collect(modules=(lin, Holder()), device_type="cpu")
    assert sum(len(g) for g in groups) == 2       # weight and bias, each once
    res = vram_reclaim.reclaim(modules=(lin, Holder()), device="cpu", probe_windows=False)
    assert isinstance(res, dict)
