"""A resume at another Learning Rate than the paused run's (families/train.py): the saved optimizer state brings
back the old rates, and this moves them to the form's new one. Light on purpose (no tqdm or model code), so
checks/test_resume_lr.py runs on the checks' own install."""
import logging
import math

logger = logging.getLogger("fizgig.families.train")   # the trainer's own console lines


def resume_lr(optimizer, fresh, meta, box_lr, adaptive=None):
    """A resume at a Learning Rate other than the paused run's: optimizer.load_state_dict brings back the saved
    rates, so the form's new one would be ignored. fresh is each group's rate as this run built it (before the load).
    Each group's start rate ('initial_lr', which the LR scheduler counts from) becomes the new one, and a run with no
    scheduler goes on at old rate x new/old. With adaptive LR the saved rate is kept, moved inside a new Min/Max.
    The same rate as before changes nothing. Returns True if the rate was changed."""
    groups = optimizer.param_groups
    if len(groups) != len(fresh):
        return False
    if adaptive is not None:
        moved = False
        for g in groups:
            lr = min(max(float(g["lr"]), adaptive.min_lr), adaptive.max_lr)
            if lr != g["lr"]:
                logger.info(f"[resume] learning rate {float(g['lr']):.3e} -> {lr:.3e}: inside the new adaptive LR "
                            f"Min/Max ({adaptive.min_lr:.3e} to {adaptive.max_lr:.3e})")
                g["lr"], moved = lr, True
        return moved
    if all("initial_lr" in g for g in groups):
        old = [float(g["initial_lr"]) for g in groups]
    elif meta.get("learning_rate") is not None and box_lr:
        old = [float(f) * float(meta["learning_rate"]) / float(box_lr) for f in fresh]
    else:
        return False                    # nothing says what the paused run started at
    if all(math.isclose(o, float(f), rel_tol=1e-9, abs_tol=0.0) for o, f in zip(old, fresh)):
        return False
    for g, o, f in zip(groups, old, fresh):
        if "initial_lr" in g:
            g["initial_lr"] = float(f)
        g["lr"] = float(g["lr"]) * float(f) / o if o > 0 else float(f)
    logger.info(f"[resume] learning rate changed: {old[0]:.3e} -> {float(fresh[0]):.3e} (the Learning Rate box); "
                f"training goes on at the new rate" + (", the LR schedule from where it paused"
                                                      if "initial_lr" in groups[0] else ""))
    return True
