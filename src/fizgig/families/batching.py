"""Batch Size > 1 for the family trainer (families/train.py): which runs may batch, and the fallback to 1.

A family opts in with FamilyDescription.supports_batch (its training_loss draws noise and a timestep per image, and
its DiT takes a stack of same-size images whose conditioning the dataset pads, `<key>__valid` marking the real rows).
Everything else - and sliders, edit pairs and full fine-tunes on any family - trains at Batch Size 1, with the reason
logged rather than a stop. Kept apart from train.py so the checks can run it without the trainer's dependencies."""
import logging

logger = logging.getLogger(__name__)


def _rebatch(ds, n):
    """A dataset's batches rebuilt at `n` images (its buckets and timestep buckets as they were)."""
    from fizgig.dataset.image_dataset import BucketBatchManager
    bm = ds.batch_manager
    ds.batch_size = n
    ds.batch_manager = BucketBatchManager(bm.buckets, n, num_timestep_buckets=bm.num_timestep_buckets)


def batch_plan(desc, group, *, slider=False, finetune=False):
    """The run's Batch Size: the dataset config's, on a family that supports batching (description supports_batch)
    and a standard LoRA run; otherwise every dataset falls back to 1, with the reason logged (never a stop)."""
    sizes = [int(getattr(ds, "batch_size", 1) or 1) for ds in getattr(group, "datasets", [])]
    n = max(sizes or [1])
    if n <= 1:
        return 1
    why = (f"{desc.display_name} trains one image at a time" if not getattr(desc, "supports_batch", False) else
           "a slider trains at Batch Size 1" if slider else
           "a full fine-tune trains at Batch Size 1" if finetune else
           "edit pairs train at Batch Size 1" if any(getattr(ds, "has_control", False) for ds in group.datasets)
           else "")
    if why:
        logger.warning(f"[batch] Batch Size {n} -> 1: {why}. Gradient Accumulation gives a larger effective batch "
                       "at the memory of batch 1.")
        for ds in group.datasets:
            if int(getattr(ds, "batch_size", 1) or 1) != 1:
                _rebatch(ds, 1)
        if hasattr(group, "cumulative_sizes"):        # a ConcatDataset caches its parts' lengths
            group.cumulative_sizes = group.cumsum(group.datasets)
        return 1
    if len(set(sizes)) > 1:
        for ds in group.datasets:
            if int(getattr(ds, "batch_size", 1) or 1) != n:
                _rebatch(ds, n)
        group.cumulative_sizes = group.cumsum(group.datasets)
    return n


# the trainer's switches that assume one image (one loss, one caption) per step, and what the log calls them
PER_IMAGE = (("log_per_image_loss", "the per-image loss watch"), ("per_image_lr", "per-image LR"),
             ("auto_recaption", "auto-recaption"), ("warmup_look_outliers", "the look warm-up"), ("dop", "DOP"),
             ("lr_boost", "LR boost"))


def batch_features(batch_size, flags):
    """The trainer's per-image switches at this Batch Size: unchanged at 1; above 1 every one is off (False / None
    for lr_boost), with a warning naming those that were on."""
    if batch_size <= 1:
        return dict(flags)
    on = [label for key, label in PER_IMAGE if flags.get(key)]
    if on:
        logger.warning(f"[batch] Batch Size {batch_size}: {', '.join(on)} off - a step's loss is the batch's "
                       "average, not one image's")
    return {**flags, **{k: (None if k == "lr_boost" else False) for k, _ in PER_IMAGE}}
