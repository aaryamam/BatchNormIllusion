"""GA: plain gradient ascent on the forget set. BN mode: train.

Raw parameter update `p.data += lr * p.grad`; no optimizer, retain data or regularisation.
"""

import torch.nn as nn

from .common import StepBudget, assert_bn_mode, bn_mode_ctx, sgd_ascent_step

BN_MODE = "train"
DEFAULTS = {"lr": 1e-3, "steps": 500}


def unlearn_ga(model, loaders, device, lr=1e-3, steps=500, bn_mode=BN_MODE,
               log=None, **_):
    crit = nn.CrossEntropyLoss()
    history = []
    with bn_mode_ctx(model, bn_mode, log=log):
        for step, (x, y) in StepBudget(loaders["forget_train_aug"], steps):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            loss = crit(model(x), y)
            sgd_ascent_step(model, loss, lr)
            assert_bn_mode(model, bn_mode)
            if log and (step % 100 == 0 or step == 1):
                log(f"  [GA] step {step}/{steps} forget_CE={loss.item():.4f}")
            history.append({"step": step, "forget_ce": float(loss.item())})
    return model, {"history": history, "bn_mode": bn_mode,
                   "hparams": {"lr": lr, "steps": steps}}
