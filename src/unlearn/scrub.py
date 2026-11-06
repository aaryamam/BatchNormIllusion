"""SCRUB: teacher/student unlearning (Kurmanji et al.). BN mode: train.

Each outer step:
  A. one gradient-ascent step on a forget batch
  B. `retain_steps` gradient-descent steps minimising
     KL(student(x_r) || teacher(x_r)) on retain batches, weighted by kl_weight.

The teacher is a frozen copy of the original model, kept in eval().
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import StepBudget, assert_bn_mode, bn_mode_ctx, sgd_ascent_step, sgd_descent_step

BN_MODE = "train"
DEFAULTS = {"lr": 1e-3, "steps": 500, "retain_steps": 2, "kl_weight": 1.0}


def unlearn_scrub(model, loaders, device, lr=1e-3, steps=500, retain_steps=2,
                  kl_weight=1.0, bn_mode=BN_MODE, log=None, **_):
    teacher = copy.deepcopy(model).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    crit = nn.CrossEntropyLoss()
    retain_iter = iter(loaders["retain_train_aug"])
    history = []

    def next_retain():
        nonlocal retain_iter
        try:
            return next(retain_iter)
        except StopIteration:
            retain_iter = iter(loaders["retain_train_aug"])
            return next(retain_iter)

    with bn_mode_ctx(model, bn_mode, log=log):
        for step, (xf, yf) in StepBudget(loaders["forget_train_aug"], steps):
            # A: ascent on forget
            xf = xf.to(device, non_blocking=True)
            yf = yf.to(device, non_blocking=True)
            loss_f = crit(model(xf), yf)
            sgd_ascent_step(model, loss_f, lr)

            # B: KL descent toward the frozen teacher on retain
            kl_val = float("nan")
            for _ in range(retain_steps):
                xr, _yr = next_retain()
                xr = xr.to(device, non_blocking=True)
                with torch.no_grad():
                    t_logits = teacher(xr)
                s_logits = model(xr)
                kl = F.kl_div(F.log_softmax(s_logits, 1),
                              F.softmax(t_logits, 1), reduction="batchmean")
                sgd_descent_step(model, kl_weight * kl, lr)
                kl_val = float(kl.item())

            assert_bn_mode(model, bn_mode)
            if log and (step % 100 == 0 or step == 1):
                log(f"  [SCRUB] step {step}/{steps} forget_CE={loss_f.item():.4f} "
                    f"KL={kl_val:.4f}")
            history.append({"step": step, "forget_ce": float(loss_f.item()), "kl": kl_val})

    return model, {"history": history, "bn_mode": bn_mode,
                   "hparams": {"lr": lr, "steps": steps, "retain_steps": retain_steps,
                               "kl_weight": kl_weight}}
