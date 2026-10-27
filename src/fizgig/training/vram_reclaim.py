"""Reclaim VRAM: move a training run's memory back into the card after Windows put part of it in system RAM.

Why: on Windows (WDDM) NVIDIA drivers 536.40+ have "CUDA - Sysmem Fallback": when the card is full (another app -
a browser playing video - took some), new CUDA allocations land in shared system memory instead of failing
(https://nvidia.custhelp.com/app/answers/detail/a_id/5490). The run carries on, several times slower (GPU busy,
low power - it waits on PCIe). Closing the other app does not bring the speed back: nothing moves those
allocations back to the card, and PyTorch's caching allocator keeps reusing the segments it already holds
(empty_cache only returns segments with nothing alive in them - torch.cuda.memory docs).

What a reclaim does, between two optimizer steps (no graph alive, nothing in flight):
  1. waits for the block swapper's transfers, then gc + synchronize + empty_cache (cached segments go back,
     wherever they sat);
  2. copies every whole CUDA tensor the run keeps between steps - the model's resident weights and buffers
     (swapped-out blocks are on the CPU and not touched), the LoRA and its grads, the optimizer's state, the
     EMA's shadow, a ring offloader's buffers - to the CPU, frees the GPU copy, empties the cache, and copies it
     back. The copy back is a fresh cudaMalloc, which the driver puts in VRAM when the card has room again.
     `t.data = ...` keeps every tensor/Parameter object: the optimizer's state (keyed by Parameter), EMA, the
     swapper and the modules all still hold the same objects. Bit-for-bit the same values: it is a copy.
  Views and tensors sharing storage with something else are left alone (moving one would break the sharing);
  tensors that alias the same whole storage are moved together and stay aliased.

There is no CUDA/PyTorch API that says an allocation went to system memory. The auto trigger uses the run's own
step time (a sustained >= 1.5x slowdown against the run's best speed, so a slow creep counts too) and waits for
torch.cuda.mem_get_info to show the card with RECLAIM_FREE_GB free again before it reclaims - reclaiming while the
card is still full would only put it back in system RAM.
"""
import gc
import logging
import os
import shutil
import subprocess
import sys
import time
from collections import deque
from statistics import median

import torch

logger = logging.getLogger(__name__)

GB = 1024 ** 3


# ---------------------------------------------------------------------------------------------------- trigger

# Free VRAM the auto reclaim waits for, the same on every card: once the whole card (every process, from
# torch.cuda.mem_get_info) shows this much free, a fresh allocation lands in VRAM again rather than in RAM.
RECLAIM_FREE_GB = 0.1


def reclaim_headroom(total=None, min_free_gb=RECLAIM_FREE_GB):
    """Bytes of device-wide free VRAM the auto reclaim waits for. Flat on every card size (total is accepted and
    ignored, so callers need not care); a Preferences value above 0 replaces the default."""
    try:
        gb = float(min_free_gb)
    except (TypeError, ValueError):
        gb = RECLAIM_FREE_GB
    return int((gb if gb > 0 else RECLAIM_FREE_GB) * GB)


class SlowdownWatch:
    """Decides when the auto reclaim runs. Feed it each clean optimizer step's time (observe); while it is `slow`,
    ask check(free, total) at each step boundary. All counts are in clean steps (steps that did sampling, saving,
    validation or epoch-boundary work are left out by the caller).

    Slow: the median of the last slow_steps steps is >= ratio x the usual step. The usual step is the fastest the
    run has been since its warm-up (the lowest median of recent normal steps), so a slow creep - 7 s, 8 s, ...
    35 s - can't drag the baseline up with it, and one quick step (a small bucket) doesn't reset the count."""

    def __init__(self, ratio=1.5, slow_steps=20, warmup_steps=50, cooldown=200, max_tries=3, window=100,
                 eval_steps=20, recover_steps=5, min_free_gb=RECLAIM_FREE_GB):
        self.ratio, self.slow_steps, self.warmup_steps = float(ratio), int(slow_steps), int(warmup_steps)
        self.cooldown, self.max_tries, self.eval_steps = int(cooldown), int(max_tries), int(eval_steps)
        self.recover_steps, self.min_free_gb = int(recover_steps), min_free_gb
        self.normal = deque(maxlen=int(window))   # recent normal step times
        self.best = None                          # the lowest median of `normal` seen after the warm-up
        self.recent = deque(maxlen=max(1, self.slow_steps))   # the last slow_steps step times, whatever they were
        self.seen = 0
        self.fast_run = 0                         # normal steps in a row while slow (it got better by itself)
        self.slow = False
        self.tries = 0
        self.stopped = False                      # a reclaim didn't help: no more auto tries this run
        self.cool_until = 0
        self.evaluating = None                    # after a reclaim: {"before": slow median, "times": [...]}
        self.last_eval = None                     # the median step time the last verdict was made on
        self.told = False                         # "spilled, waiting for room" said for this slowdown

    @property
    def baseline(self):
        if not self.normal:
            return None
        cur = median(self.normal)
        return cur if self.best is None else min(cur, self.best)

    @property
    def armed(self):
        return (not self.stopped and self.tries < self.max_tries and self.seen >= self.warmup_steps
                and self.seen >= self.cool_until and len(self.normal) >= min(10, self.warmup_steps or 10))

    def _note_normal(self, dt):
        self.normal.append(dt)
        if self.seen > self.warmup_steps and len(self.normal) >= min(10, self.normal.maxlen):
            m = median(self.normal)
            self.best = m if self.best is None else min(self.best, m)

    def observe(self, dt):
        """One clean step's seconds. Returns an event: "slow" (a sustained slowdown just confirmed), "recovered"
        (it went back to normal by itself), "improved" / "no_improvement" (the verdict on the last reclaim),
        or None."""
        self.seen += 1
        base = self.baseline
        if self.evaluating is not None:
            self.evaluating["times"].append(dt)
            if len(self.evaluating["times"]) < self.eval_steps:
                return None
            after = median(self.evaluating["times"])
            before, manual = self.evaluating["before"], self.evaluating["manual"]
            self.evaluating, self.last_eval = None, after
            ok = (base is not None and after < self.ratio * base) or bool(before and after <= 0.8 * before)
            if ok:
                return "improved"
            if not manual:
                self.stopped = True               # an automatic one that didn't help: no more automatic tries
            return "no_improvement"
        if self.seen <= self.warmup_steps or base is None:
            self.normal.append(dt)                 # the warm-up: the median follows the run as it speeds up
            return None
        self.recent.append(dt)
        if dt < self.ratio * base:
            self._note_normal(dt)
            if self.slow:
                self.fast_run += 1
                if self.fast_run >= self.recover_steps:
                    self._calm()
                    return "recovered"
            return None
        self.fast_run = 0
        if (not self.slow and len(self.recent) >= self.slow_steps and median(self.recent) >= self.ratio * base
                and self.armed):
            self.slow, self.told = True, False
            return "slow"
        return None

    def _calm(self):
        self.slow, self.fast_run, self.told = False, 0, False
        self.recent.clear()

    def check(self, free, total=None):
        """While slow: "reclaim" once the whole card shows reclaim_headroom() free (free None = can't tell, so
        reclaim); "wait_log" the first time it has to wait in this slowdown (say so once), then "wait" for as long
        as the card stays full - no step limit, a reclaim into a full card would only land in RAM again. None
        when not slow."""
        if not self.slow or not self.armed:
            return None
        if free is None or free >= reclaim_headroom(total, self.min_free_gb):
            return "reclaim"
        if not self.told:
            self.told = True
            return "wait_log"
        return "wait"

    def reclaimed(self, manual=False):
        """A reclaim ran: judge it over the next eval_steps clean steps; auto ones count toward max_tries."""
        before = median(self.recent) if self.recent else None
        if not manual:
            self.tries += 1
        self._calm()
        self.cool_until = self.seen + self.cooldown
        self.evaluating = {"before": before, "times": [], "manual": bool(manual)}


def vram_free_total(device=None):
    """(free, total) bytes for the whole card - every process, not just this run's allocator - or (None, None)
    when torch can't tell."""
    try:
        free, total = torch.cuda.mem_get_info(device)
        return int(free), int(total)
    except Exception:
        return None, None


# ---------------------------------------------------------------------------------------------------- reclaim

def _walk_tensors(obj, out, depth=0):
    if torch.is_tensor(obj):
        out.append(obj)
    elif depth < 4 and isinstance(obj, dict):
        for v in obj.values():
            _walk_tensors(v, out, depth + 1)
    elif depth < 4 and isinstance(obj, (list, tuple)):
        for v in obj:
            _walk_tensors(v, out, depth + 1)


def find_offloaders(modules):
    """Block-swap offloaders hanging off the modules (model.offloader, offloader_double, ...)."""
    found, seen = [], set()
    for m in modules:
        if m is None or not hasattr(m, "modules"):
            continue
        for sub in m.modules():
            for k, v in vars(sub).items():
                if "offloader" in k and v is not None and id(v) not in seen and not isinstance(v, torch.nn.Module):
                    seen.add(id(v))
                    found.append(v)
    return found


def settle_offloaders(offloaders):
    """Let every pending block transfer finish, so no swap is half-done while tensors move."""
    for off in offloaders:
        futs = getattr(off, "futures", None)
        if isinstance(futs, dict) and hasattr(off, "_wait_blocks_move"):
            for k in list(futs.keys()):
                off._wait_blocks_move(k)
        cop = getattr(off, "copier", None)
        if cop is not None and hasattr(cop, "sync"):
            cop.sync()


def _whole(t):
    """The tensor owns its whole storage (not a view into something bigger, nothing else packed beside it)."""
    try:
        if t._base is not None or t.storage_offset() != 0:
            return False
        return t.untyped_storage().nbytes() == t.numel() * t.element_size()
    except Exception:
        return False


def _ring_jobs(offloaders, device_type):
    """LoRAStreamOffloader's GPU ring: flat byte buffers whose ring Parameters are views. Moved as one: the flat
    buffer is copied, each ring Parameter re-pointed at the same offset of the new one."""
    jobs = []
    for off in offloaders:
        flats, params = getattr(off, "ring_flat", None), getattr(off, "ring_param", None)
        if not flats or not params:
            continue
        for slot, flat in enumerate(flats):
            if flat.device.type != device_type:
                continue
            base = flat.data_ptr()
            views = [(p, p.data.data_ptr() - base, p.data.dtype, tuple(p.data.shape)) for p in params[slot]]
            jobs.append((off, slot, views))
    return jobs


def collect(modules=(), optimizers=(), extra=(), device_type="cuda", exclude=()):
    """Groups of tensors to move: [[tensor, ...]] - each group shares one whole storage (usually one tensor)."""
    cand = []
    for m in modules:
        if m is None:
            continue
        for p in (m.parameters() if hasattr(m, "parameters") else ()):
            cand.append(p)
            if p.grad is not None:
                cand.append(p.grad)
        # not every holder is an nn.Module: the families' FamilyLoRA only has parameters() (its adapters live
        # inside the DiT, which is passed too)
        for b in (m.buffers() if hasattr(m, "buffers") else ()):
            cand.append(b)
    for opt in optimizers:
        if opt is None:
            continue
        for st in getattr(opt, "state", {}).values():
            _walk_tensors(st, cand)
        for g in getattr(opt, "param_groups", []):
            _walk_tensors({k: v for k, v in g.items() if k != "params"}, cand)
    for e in extra:
        _walk_tensors(e, cand)
    groups, by_ptr, seen, skipped = [], {}, set(exclude), 0
    for t in cand:
        if id(t) in seen:
            continue
        seen.add(id(t))
        if t.device.type != device_type or t.numel() == 0:
            continue
        if getattr(t, "is_paged", False) or type(t) not in (torch.Tensor, torch.nn.Parameter):
            skipped += 1                           # paged (bitsandbytes) or a tensor subclass: left alone
            continue
        ptr = t.data_ptr()
        g = by_ptr.get(ptr)
        if g is None:
            by_ptr[ptr] = g = {"ok": True, "ts": []}
            groups.append(g)
        g["ts"].append(t)
    out = []
    for g in groups:
        t0 = g["ts"][0]
        same = all(t.dtype == t0.dtype and t.shape == t0.shape and t.stride() == t0.stride() for t in g["ts"])
        if same and all(_whole(t) for t in g["ts"]):
            out.append(g["ts"])
        else:
            skipped += len(g["ts"])
    return out, skipped


def _host_free():
    try:
        import psutil
        return psutil.virtual_memory().available
    except Exception:
        return None


def windows_gpu_process_memory(pid=None):
    """(dedicated, shared) GPU memory bytes of this process from Windows' GPU Process Memory counters, or None.
    Shared includes pinned host buffers (block swap's), so compare before / after rather than reading it alone."""
    if sys.platform != "win32" or not shutil.which("typeperf"):
        return None
    pid = pid or os.getpid()
    try:
        out = subprocess.run(["typeperf", f"\\GPU Process Memory(pid_{pid}_*)\\Dedicated Usage",
                              f"\\GPU Process Memory(pid_{pid}_*)\\Shared Usage", "-sc", "1"],
                             capture_output=True, text=True, timeout=15,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        lines = [ln for ln in out.stdout.splitlines() if ln.startswith('"')]
        head, row = lines[0].split('","'), lines[1].split('","')
        ded = shr = 0.0
        for h, v in zip(head[1:], row[1:]):
            try:
                val = float(v.strip('"'))
            except ValueError:
                continue
            if "Dedicated" in h:
                ded += val
            elif "Shared" in h:
                shr += val
        return int(ded), int(shr)
    except Exception:
        return None


def _mem(device_type):
    if device_type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        free, total = torch.cuda.mem_get_info()
        return {"free": free, "total": total, "reserved": torch.cuda.memory_reserved(),
                "allocated": torch.cuda.memory_allocated()}
    except Exception:
        return None


def _flush(device_type):
    gc.collect()
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


@torch.no_grad()
def reclaim(modules=(), optimizers=(), extra=(), device="cuda", staging="cpu", chunk_bytes=None, probe_windows=True):
    """Move the run's GPU tensors out and back (see the module doc). Call between optimizer steps only.
    Returns a dict of what it did and the memory before / after."""
    device = torch.device(device)
    dt = device.type
    t_start = time.time()
    modules = [m for m in modules if m is not None]
    offs = find_offloaders(modules)
    settle_offloaders(offs)
    _flush(dt)
    before = _mem(dt)
    win_before = windows_gpu_process_memory() if probe_windows else None

    rings = _ring_jobs(offs, dt)
    ring_ids = {id(p) for _o, _s, views in rings for p, *_ in views}
    ring_flat_ids = {id(off.ring_flat[s]) for off, s, _v in rings}
    groups, skipped = collect(modules, optimizers, extra, dt, exclude=ring_ids | ring_flat_ids)
    total = sum(g[0].numel() * g[0].element_size() for g in groups)
    total += sum(off.ring_flat[s].numel() for off, s, _v in rings)
    if chunk_bytes is None:
        avail = _host_free()
        chunk_bytes = total if avail is None or total < 0.6 * avail else max(256 * 1024 ** 2, int(0.5 * avail))

    def _move(group_list, ring_list):
        held = []
        for g in group_list:                   # out: a CPU copy; the GPU storage is let go
            cpu = g[0].detach().to(staging, copy=True)
            for t in g:
                t.data = cpu
            held.append((g, cpu))
        ring_held = []
        for off, s, views in ring_list:
            cpu = off.ring_flat[s].to(staging, copy=True)
            off.ring_flat[s] = cpu
            ring_held.append((off, s, views, cpu))
        _flush(dt)                              # the old segments go back to the driver
        for g, cpu in held:                    # back: a fresh allocation - VRAM when the card has room
            new = cpu.to(device, copy=True)
            for t in g:
                t.data = new
        for off, s, views, cpu in ring_held:
            new = cpu.to(device, copy=True)
            off.ring_flat[s] = new
            for p, offset, dtype, shape in views:
                n = 1
                for x in shape:
                    n *= x
                nbytes = n * torch.empty((), dtype=dtype).element_size()
                p.data = new[offset:offset + nbytes].view(dtype).view(shape)

    chunk, size = [], 0
    for g in groups:
        chunk.append(g)
        size += g[0].numel() * g[0].element_size()
        if size >= chunk_bytes:
            _move(chunk, [])
            chunk, size = [], 0
    _move(chunk, rings)
    _flush(dt)
    return {"tensors": sum(len(g) for g in groups) + sum(len(v) for _o, _s, v in rings), "bytes": total,
            "skipped": skipped, "seconds": time.time() - t_start, "before": before, "after": _mem(dt),
            "win_before": win_before, "win_after": windows_gpu_process_memory() if probe_windows else None}


def describe(res):
    """One short console line for a reclaim: what moved, and Windows' shared GPU memory before -> after when it
    can be read (the number that says whether it went back into VRAM), else free VRAM before -> after."""
    gb = lambda b: f"{b / GB:.1f}"
    s = f"[vram] Moved {gb(res['bytes'])} GB of training memory out and back in {res['seconds']:.0f} s"
    wb, wa = res.get("win_before"), res.get("win_after")
    b, a = res.get("before"), res.get("after")
    if wb and wa:
        s += f" (shared GPU memory {gb(wb[1])} -> {gb(wa[1])} GB)"
    elif b and a:
        s += f" (free VRAM {gb(b['free'])} -> {gb(a['free'])} GB)"
    return s + "."
