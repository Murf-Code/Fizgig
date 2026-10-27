"""Early stopping on the fixed validation loss (training/val_loss.py): the run finishes once the validation loss has
not improved for `patience` checks in a row, and its best check's LoRA becomes the final one.

"Improved" means lower than the best so far by more than a small tolerance - `min_delta`, a fraction of the best
(0.001 = 0.1%) - so noise in the last digits doesn't count as progress. A resume carries the count on (its state is
saved with the run's resume state); a run continued after it stopped early starts counting afresh, and a resume that
changed what validation checks (a new section of the validation line) starts a new best.
"""

PATIENCE = 3              # checks without a better validation loss before the run stops
MIN_DELTA = 0.001         # what counts as better: below the best by more than 0.1% of it


class EarlyStop:
    def __init__(self, patience=PATIENCE, min_delta=MIN_DELTA):
        self.patience = max(1, int(patience))
        self.min_delta = max(0.0, float(min_delta))
        self.best = None          # the best validation loss so far
        self.best_step = None     # its step and epoch
        self.best_epoch = None
        self.bad = 0              # checks in a row since then without a better one
        self.seg = 0              # the validation line's section the best belongs to
        self.stopped = False
        self.best_saved = False   # the best check's LoRA is on disk (name-best.safetensors)

    def observe(self, v, step, epoch, seg=0):
        """One check's loss -> "best" (a new best: keep its LoRA), "wait" (not better yet) or "stop"."""
        v = float(v)
        if seg != self.seg:                    # other images / noise now: the old best isn't comparable
            self.best, self.bad, self.seg, self.best_saved = None, 0, seg, False
        if self.best is None or v < self.best - self.min_delta * abs(self.best):
            self.best, self.best_step, self.best_epoch, self.bad = v, int(step), int(epoch), 0
            self.best_saved = False
            return "best"
        self.bad += 1
        if self.bad >= self.patience:
            self.stopped = True
            return "stop"
        return "wait"

    def state_dict(self):
        return {k: getattr(self, k) for k in ("best", "best_step", "best_epoch", "bad", "seg", "stopped",
                                              "best_saved")}

    def load_state_dict(self, d):
        """A resume's count back. One that had stopped early (a Continue for more epochs) counts afresh from its
        best; False then, else True."""
        if not isinstance(d, dict):
            return True
        for k in ("best", "best_step", "best_epoch", "bad", "seg", "best_saved"):
            if k in d:
                setattr(self, k, d[k])
        if d.get("stopped"):
            self.bad, self.stopped = 0, False
            return False
        return True
