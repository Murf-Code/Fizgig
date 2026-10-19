"""A resume at another Learning Rate: the saved optimizer state brings back the paused run's rates, so the trainer
moves them to the form's new one (and logs it); the same rate as before leaves the restored state exactly as it was."""
import logging

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.filterwarnings("ignore:Detected call of `lr_scheduler.step\(\)` before")
from fizgig.families import train as T  # noqa: E402
from fizgig.training.adaptive_lr import AdaptiveLR  # noqa: E402


def _groups(lr):
    a, b = torch.nn.Parameter(torch.ones(4)), torch.nn.Parameter(torch.ones(4))
    return [{"params": [a], "lr": lr}, {"params": [b], "lr": lr * 0.5}]    # a second group at half the rate


def _paused(lr, kind="cosine", steps=7, total=20):
    """A run at lr with a scheduler, stepped `steps` times, as its optimizer.pt would hold it."""
    opt = torch.optim.AdamW(_groups(lr))
    sch = T._step_scheduler(opt, kind, 0, total) if kind else None
    for _ in range(steps):
        opt.zero_grad()
        sum(p.sum() for g in opt.param_groups for p in g["params"]).backward()
        opt.step()
        if sch:
            sch.step()
    return opt.state_dict()


def _resume(sd, lr, kind="cosine", steps=7, total=20, meta=None, adaptive=None):
    """The trainer's order: build at the form's lr, note the fresh rates, load the state, adjust, build the scheduler
    and move it to where the run paused."""
    opt = torch.optim.AdamW(_groups(lr))
    fresh = [g["lr"] for g in opt.param_groups]
    opt.load_state_dict(sd)
    changed = T._resume_lr(opt, fresh, meta or {"learning_rate": None}, lr, adaptive)
    if kind:
        sch = T._step_scheduler(opt, kind, 0, total)
        for _ in range(steps):
            sch.step()
    return opt, changed


@pytest.mark.parametrize("kind", ["cosine", "constant"])
def test_same_rate_is_untouched(kind, caplog):
    sd = _paused(1e-4, kind)
    with caplog.at_level(logging.INFO):
        opt, changed = _resume(sd, 1e-4, kind)
    ref, _ = _resume(sd, 1e-4, kind)
    plain = torch.optim.AdamW(_groups(1e-4))           # what a resume did before this change
    plain.load_state_dict(sd)
    sch = T._step_scheduler(plain, kind, 0, 20)
    for _ in range(7):
        sch.step()
    assert not changed and "learning rate changed" not in caplog.text
    assert [g["lr"] for g in opt.param_groups] == [g["lr"] for g in plain.param_groups]
    assert [g["initial_lr"] for g in opt.param_groups] == [1e-4, 5e-5]


@pytest.mark.parametrize("kind", ["cosine", "constant"])
def test_higher_rate_takes_over(kind, caplog):
    sd = _paused(1e-4, kind)
    with caplog.at_level(logging.INFO):
        opt, changed = _resume(sd, 3e-4, kind)
    assert changed and "learning rate changed: 1.000e-04 -> 3.000e-04" in caplog.text
    fresh_run = torch.optim.AdamW(_groups(3e-4))       # a run at 3e-4 from the start, at the same point
    sch = T._step_scheduler(fresh_run, kind, 0, 20)
    for _ in range(7):
        sch.step()
    for g, w in zip(opt.param_groups, fresh_run.param_groups):
        assert g["lr"] == pytest.approx(w["lr"], rel=1e-9)
        assert g["initial_lr"] == pytest.approx(w["initial_lr"], rel=1e-9)
    assert opt.state_dict()["state"][0]["step"] == 7     # the moments came back with it


def test_no_scheduler_uses_saved_rate_record(caplog):
    sd = _paused(1e-4, None)
    with caplog.at_level(logging.INFO):
        opt, changed = _resume(sd, 2e-4, None, meta={"learning_rate": 1e-4})
    assert changed and [g["lr"] for g in opt.param_groups] == pytest.approx([2e-4, 1e-4])
    opt, changed = _resume(sd, 1e-4, None, meta={"learning_rate": 1e-4})
    assert not changed and [g["lr"] for g in opt.param_groups] == [1e-4, 5e-5]
    opt, changed = _resume(sd, 2e-4, None, meta={})     # an older state with no record: as before
    assert not changed and [g["lr"] for g in opt.param_groups] == [1e-4, 5e-5]


def test_adaptive_keeps_its_rate_inside_new_bounds(caplog):
    sd = _paused(8e-5, None)                            # the watcher had reached 8e-5 (group 2: 4e-5)
    opt, changed = _resume(sd, 1e-4, None, adaptive=AdaptiveLR(1e-5, 4e-4))
    assert not changed and [g["lr"] for g in opt.param_groups] == [8e-5, 4e-5]
    with caplog.at_level(logging.INFO):
        opt, changed = _resume(sd, 1e-4, None, adaptive=AdaptiveLR(5e-5, 4e-4))
    assert changed and [g["lr"] for g in opt.param_groups] == [8e-5, 5e-5]
    assert "inside the new adaptive LR Min/Max" in caplog.text
