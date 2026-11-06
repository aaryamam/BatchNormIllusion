"""IncompTeacher: competent/incompetent teacher pair (Chundawat et al.). BN mode: train.

Each step draws one forget batch and one retain batch:

    L = alpha * KL( student(x_f)/T || incompetent(x_f)/T )
      + (1 - alpha) * KL( student(x_r)/T || competent(x_r)/T )

`incompetent` is a freshly initialised network and `competent` a frozen copy of the model
before unlearning. KL is scaled by T^2.
"""

import copy

import torch
import torch.nn.functional as F

from .common import (StepBudget, assert_bn_mode, bn_mode_ctx, random_init_like,
                     sgd_descent_step)

BN_MODE = "train"
DEFAULTS = {"lr": 1e-3, "steps": 500, "temperature": 4.0, "alpha": 0.95}


def _kl(student_logits, teacher_logits, T):
    return F.kl_div(F.log_softmax(student_logits / T, 1),
                    F.softmax(teacher_logits / T, 1),
                    reduction="batchmean") * (T * T)


def unlearn_incomp_teacher(model, loaders, device, lr=1e-3, steps=500,
                           temperature=4.0, alpha=0.95, bn_mode=BN_MODE,
                           seed=0, log=None, **_):
    competent = copy.deepcopy(model).to(device).eval()
    for p in competent.parameters():
        p.requires_grad_(False)

    torch.manual_seed(seed + 12345)
    incompetent = random_init_like(model).to(device).eval()
    for p in incompetent.parameters():
        p.requires_grad_(False)

    retain_iter = iter(loaders["retain_train_aug"])

    def next_retain():
        nonlocal retain_iter
        try:
            return next(retain_iter)
        except StopIteration:
            retain_iter = iter(loaders["retain_train_aug"])
            return next(retain_iter)

    history = []
    with bn_mode_ctx(model, bn_mode, log=log):
        for step, (xf, _yf) in StepBudget(loaders["forget_train_aug"], steps):
            xf = xf.to(device, non_blocking=True)
            xr, _yr = next_retain()
            xr = xr.to(device, non_blocking=True)

            with torch.no_grad():
                t_bad = incompetent(xf)
                t_good = competent(xr)

            loss_f = _kl(model(xf), t_bad, temperature)
            loss_r = _kl(model(xr), t_good, temperature)
            loss = alpha * loss_f + (1.0 - alpha) * loss_r

            sgd_descent_step(model, loss, lr)
            assert_bn_mode(model, bn_mode)

            if log and (step % 100 == 0 or step == 1):
                log(f"  [IncompTeacher] step {step}/{steps} "
                    f"L={loss.item():.4f} (f={loss_f.item():.4f} r={loss_r.item():.4f})")
            history.append({"step": step, "loss": float(loss.item()),
                            "kl_forget": float(loss_f.item()),
                            "kl_retain": float(loss_r.item())})

    return model, {"history": history, "bn_mode": bn_mode,
                   "hparams": {"lr": lr, "steps": steps, "temperature": temperature,
                               "alpha": alpha,
                               "alpha_weights": "forget/incompetent term"}}
