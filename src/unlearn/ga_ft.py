"""GA+FT: gradient ascent, then a short retain fine-tune. BN mode: train.

GA for 200 steps at lr=1e-3, then 1 epoch of retain fine-tuning at lr=1e-4. The fine-tune
runs with BN in train mode on retain data, so it recalibrates the running statistics as a
side effect.
"""

import torch.nn as nn

from .common import StepBudget, assert_bn_mode, bn_mode_ctx, sgd_ascent_step, sgd_descent_step

BN_MODE = "train"
DEFAULTS = {"ga_lr": 1e-3, "ga_steps": 200, "ft_lr": 1e-4, "ft_epochs": 1}


def unlearn_ga_ft(model, loaders, device, ga_lr=1e-3, ga_steps=200, ft_lr=1e-4,
                  ft_epochs=1, bn_mode=BN_MODE, seed=0, log=None, **_):
    crit = nn.CrossEntropyLoss()
    history = []

    with bn_mode_ctx(model, bn_mode, log=log):
        # ---- Phase 1: gradient ascent on the forget set --------------------
        for step, (x, y) in StepBudget(loaders["forget_train_aug"], ga_steps):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            loss = crit(model(x), y)
            sgd_ascent_step(model, loss, ga_lr)
            assert_bn_mode(model, bn_mode)
            if log and (step % 50 == 0 or step == 1):
                log(f"  [GA+FT/ga] step {step}/{ga_steps} forget_CE={loss.item():.4f}")
            history.append({"phase": "ga", "step": step, "forget_ce": float(loss.item())})

        # ---- Phase 2: retain fine-tune, BN still in train mode -------------
        ft_step = 0
        for ep in range(ft_epochs):
            for x, y in loaders["retain_train_aug"]:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                loss = crit(model(x), y)
                sgd_descent_step(model, loss, ft_lr)
                assert_bn_mode(model, bn_mode)
                ft_step += 1
                if log and (ft_step % 100 == 0 or ft_step == 1):
                    log(f"  [GA+FT/ft] epoch {ep+1}/{ft_epochs} step {ft_step} "
                        f"retain_CE={loss.item():.4f}")
                history.append({"phase": "ft", "step": ft_step,
                                "retain_ce": float(loss.item())})

    return model, {"history": history, "bn_mode": bn_mode,
                   "n_ft_steps": ft_step,
                   "hparams": {"ga_lr": ga_lr, "ga_steps": ga_steps, "ft_lr": ft_lr,
                               "ft_epochs": ft_epochs}}
