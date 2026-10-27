"""Fixed validation loss with block swap (Chris's Krea 2 run, int8 + 24 of 28 blocks swapped): the end-of-epoch check
crashed with "mat2 is on cpu". A forward with no backward, in the offloader's training layout, moves each early block to
the CPU and counts on the backward hooks to bring it back - so the check's second forward met block 0 on the CPU.

On CPU this drives the real Krea 2 offloader (fizgig.krea2.offloading.ModelOffloader: its forward schedule, backward
hooks and forward-only switch) and the real SingleStreamDiT swap switches, with only the device moves simulated: each
block carries where its weights are, and a block whose weights are "on the CPU" fails as the int8 matmul did."""
import logging
import os
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fizgig.krea2 import offloading  # noqa: E402
from fizgig.krea2.model import SingleStreamDiT  # noqa: E402
from fizgig.training import val_loss  # noqa: E402
from fizgig.training.val_loss import FixedValidation, run_or_skip  # noqa: E402

N_BLOCKS, SWAP = 8, 6            # most blocks streamed, as on a 12 GB card (24 of 28)


class Block(nn.Module):
    def __init__(self, i):
        super().__init__()
        torch.manual_seed(i)
        self.lin = nn.Linear(4, 4)
        self.where = "gpu"

    def forward(self, x):
        if self.where != "gpu":
            raise RuntimeError("Expected all tensors to be on the same device, but got mat2 is on cpu")
        return x + 0.1 * torch.tanh(self.lin(x))


class TinyDiT(nn.Module):
    """Krea 2's block loop (model.py SingleStreamDiT.forward) and its own swap switches."""
    switch_block_swap_for_inference = SingleStreamDiT.switch_block_swap_for_inference
    switch_block_swap_for_training = SingleStreamDiT.switch_block_swap_for_training
    prepare_block_swap_before_forward = SingleStreamDiT.prepare_block_swap_before_forward

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(Block(i) for i in range(N_BLOCKS))
        self.lora = nn.Parameter(torch.zeros(4))          # the trainable part (the LoRA)
        self.blocks_to_swap = SWAP
        self.offloader = offloading.ModelOffloader("single", self.blocks, N_BLOCKS, SWAP, True, torch.device("cpu"))
        self.prepare_block_swap_before_forward()

    def forward(self, x):
        x = x + self.lora
        for index, block in enumerate(self.blocks):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(index)
            x = block(x)
            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(self.blocks, index)
        return x


class Driver:
    """The family driver's side: Krea 2's block_swap_mode, and a training loss from the model's forward."""
    from fizgig.krea2.driver import Krea2Driver
    block_swap_mode = Krea2Driver.block_swap_mode

    def batch_cond(self, batch, device):
        return None

    def training_loss(self, dit, latents, cond, g, min_t=0.0, max_t=1.0, **_):
        noise = torch.randn(latents.shape, generator=g)
        return ((dit(latents + min_t * noise) - latents) ** 2).mean(), None


@pytest.fixture
def sim(monkeypatch):
    """Device moves simulated on the blocks' `where` (the offloader's scheduling stays its own)."""
    def swap(self, to_cpu, to_gpu):
        to_cpu.where, to_gpu.where = "cpu", "gpu"

    def prepare(self, blocks):
        for i, b in enumerate(blocks):
            b.where = "gpu" if i < self.num_blocks - self.blocks_to_swap else "cpu"

    monkeypatch.setattr(offloading.Offloader, "swap_weight_devices", swap)
    monkeypatch.setattr(offloading.ModelOffloader, "prepare_block_devices_before_forward", prepare)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    return TinyDiT(), Driver()


def _layout(dit):
    return [b.where for b in dit.blocks]


def _train_step(dit):
    loss = ((dit(torch.ones(2, 4)) - 1) ** 2).mean()
    loss.backward()
    dit.offloader.wait_for_block(0)          # settle the last move the backward hooks submitted
    return loss.item()


def _validator():
    v = FixedValidation(["a", "b"], images=2, timesteps=3, seed=42)
    for k in ("a", "b"):
        v.offer({"item_keys": [k], "latents": torch.full((1, 4), 0.5 if k == "a" else -0.5)})
    return v


def test_training_steps_alone_keep_the_layout(sim):
    dit, _ = sim
    start = _layout(dit)
    for _ in range(3):
        _train_step(dit)
    assert _layout(dit) == start


def test_old_path_reproduces_the_crash(sim):
    """The check as it ran before (no forward-only mode): the second forward finds block 0 on the CPU."""
    dit, drv = sim
    _train_step(dit)
    with pytest.raises(RuntimeError, match="mat2 is on cpu"):
        _validator().run(drv, dit, "cpu", swapped=False)


def test_validation_streams_as_inference_and_training_carries_on(sim):
    dit, drv = sim
    start = _layout(dit)
    _train_step(dit)
    r = _validator().run(drv, dit, "cpu", swapped=True)
    assert r is not None and len(r[1]) == 3
    assert dit.offloader.forward_only is False                  # back in the training mode
    assert _layout(dit) == start                                # every block where training expects it
    for _ in range(2):                                          # and training goes on as before
        _train_step(dit)
    again = _validator().run(drv, dit, "cpu", swapped=True)
    assert again[0] == pytest.approx(r[0])                      # nothing trained in between: the same number
    assert _layout(dit) == start


def test_a_failed_check_never_stops_training(sim, caplog):
    dit, drv = sim
    start = _layout(dit)
    _train_step(dit)
    v = _validator()

    def boom(*a, **k):
        raise RuntimeError("anything at all")
    drv.training_loss = boom
    with caplog.at_level(logging.DEBUG, logger=val_loss.logger.name):
        r, failed = run_or_skip(v, drv, dit, "cpu", swapped=True)
    assert r is None and failed
    notes = [x for x in caplog.records if x.levelno == logging.WARNING]
    assert len(notes) == 1 and "Training carries on" in notes[0].getMessage()
    assert any(x.levelno == logging.DEBUG and x.exc_info for x in caplog.records)     # the details, in the log
    assert dit.offloader.forward_only is False and _layout(dit) == start
    _train_step(dit)                                            # training goes on


def test_trainer_turns_a_failed_check_off():
    src = open(os.path.join(os.path.dirname(__file__), "..", "src", "fizgig", "families", "train.py"),
               encoding="utf-8").read()
    body = src[src.index("    def _validate(epoch_no):"):src.index("    share_skip = [0]")]
    assert "nonlocal validator" in body and "run_or_skip(" in body and "swapped=bool(swapped)" in body
    assert "if failed:\n            validator = None" in body
    assert "validator.run(" not in body                          # never called bare (an error would end the run)
