"""With the fixed validation loss on, Adaptive LR judges on it (smooth) instead of the jittery train loss, and after a
halving it waits for fresh evidence: Chris's Muon Krea 2 run (29 Oct 2026) floored at Min by epoch 16 on train loss."""
import torch

from fizgig.training.adaptive_lr import AdaptiveLR

VAL = [.13597, .13542, .13549, .13552, .13590, .13491, .13486, .13443, .13394, .13442, .13405, .13426, .13401,
       .13405, .13384, .13376, .13365, .13361, .13358, .13358, .13348, .13362, .13346]
TRAIN = [.1431, .1421, .1404, .1379, .1428, .1309, .1446, .1450, .1361, .1389, .1306, .1232, .1422, .1377, .1327,
         .1329, .1383, .1344, .1413, .1376, .1303, .1381, .1356]
START, MIN, MAX = 8.9e-5, 1e-5, 2e-4


def _replay(train, val, a=None):
    net = torch.nn.Linear(4, 4)
    opt = torch.optim.SGD(net.parameters(), lr=START)
    a = a or AdaptiveLR(MIN, MAX)
    lrs = []
    for e, t in enumerate(train):
        a.epoch_boundary(e, t, net, opt, val_loss=None if val is None else val[e])
        lrs.append(opt.param_groups[0]["lr"])
    return a, lrs


def test_validation_loss_keeps_the_lr_well_above_min():
    _a, lrs = _replay(TRAIN, VAL)
    print(" ".join(f"{x:.2e}" for x in lrs))
    assert min(lrs) >= 3 * MIN and lrs[-1] >= 5 * MIN


def test_a_real_plateau_still_halves_but_not_every_two_epochs():
    _a, lrs = _replay([.14] * 12, [.1335] * 12)
    cuts = [i for i in range(1, len(lrs)) if lrs[i] < lrs[i - 1]]
    assert cuts and all(b - a >= 3 for a, b in zip(cuts, cuts[1:]))


def test_without_validation_it_still_uses_the_train_loss():
    _a, lrs = _replay([.14] * 6, None)
    assert lrs[-1] < START


def test_the_val_state_survives_a_resume():
    a, _ = _replay(TRAIN[:9], VAL[:9])
    b = AdaptiveLR(MIN, MAX)
    b.load_state_dict(a.state_dict())
    assert b.best_val == a.best_val and b.cooldown == a.cooldown
