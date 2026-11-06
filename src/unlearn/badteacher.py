"""BadTeacher: competent / incompetent teacher KL (Chundawat et al.). BN mode: eval.

The running statistics inherited from the original model are used in the forward pass
and never updated. On forget samples the student is distilled toward a randomly
initialised teacher; on retain samples toward a frozen copy of the original model.

`retain_fraction` controls the composition of the unlearning loader (0.0 = forget-only).
"""

import copy

import torch
import torch.nn.functional as F

from ..data.cifar import make_mixed_unlearn_loader
from .common import (StepBudget, assert_bn_mode, bn_mode_ctx, random_init_like,
                     sgd_descent_step)

BN_MODE = "eval"
# BadTeacher and IncompTeacher differ only in BN mode at these settings. The original
# Adam/1e-4/one-epoch configuration is available through overrides.
DEFAULTS = {"lr": 1e-3, "steps": 500, "temperature": 4.0, "alpha": 0.95,
            "retain_fraction": 0.0, "optimizer": "sgd", "epochs": 1}


def _kl(student_logits, teacher_logits, T=1.0):
    return F.kl_div(F.log_softmax(student_logits / T, 1),
                    F.softmax(teacher_logits / T, 1),
                    reduction="batchmean") * (T * T)


def unlearn_badteacher(model, loaders, device, lr=1e-4, steps=None, temperature=4.0,
                       alpha=0.95, retain_fraction=0.0, optimizer="adam", epochs=1,
                       bn_mode=BN_MODE, seed=0, log=None, **_):
    """`alpha` weights the forget term against the retain term.

    It applies only when retain_fraction == 0; with a mixed loader the blend comes from the
    batch composition and alpha is ignored.
    """
    competent = copy.deepcopy(model).to(device).eval()
    for p in competent.parameters():
        p.requires_grad_(False)

    torch.manual_seed(seed + 12345)
    incompetent = random_init_like(model).to(device).eval()
    for p in incompetent.parameters():
        p.requires_grad_(False)

    mixed = retain_fraction > 0.0
    batch_size = loaders["config"]["batch_size"]

    # With optimizer="adam", one epoch over the unlearning loader sets the step count.
    # With a mixed loader an "epoch" is defined by how many batches are needed to see
    # the forget set once, since the forget set is what is being unlearned.
    if steps is None:
        n_forget = loaders["config"]["n_forget_train"]
        per_batch_forget = batch_size - int(round(retain_fraction * batch_size))
        per_batch_forget = max(per_batch_forget, 1)
        steps = int(epochs * max(1, n_forget // per_batch_forget))

    if mixed:
        # yields (x, y, is_forget) indefinitely
        stream = iter(make_mixed_unlearn_loader(loaders, retain_fraction, batch_size))
    else:
        # forget-only: every sample is a forget sample
        base = StepBudget(loaders["forget_train_aug"], steps)
        stream = ((x, y, torch.ones(x.size(0), dtype=torch.bool))
                  for _s, (x, y) in base)

    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "sgd":
        opt = None                      # raw parameter update, as the other methods use
    else:
        raise ValueError(f"optimizer must be 'adam' or 'sgd', got {optimizer!r}")

    use_alpha = (not mixed) and alpha < 1.0
    retain_iter = iter(loaders["retain_train_aug"]) if use_alpha else None

    def next_retain():
        nonlocal retain_iter
        try:
            return next(retain_iter)
        except StopIteration:
            retain_iter = iter(loaders["retain_train_aug"])
            return next(retain_iter)

    history = []
    from .common import set_bn_mode
    with bn_mode_ctx(model, bn_mode, log=log):
        for step in range(1, steps + 1):
            # model.train(), then every nn.BatchNorm2d flipped to
            # .eval() INSIDE the step loop - not model.eval(). `set_bn_mode` is exactly
            # that (model.train() followed by BN-only .eval()), re-applied each step so
            # nothing in the loop body can silently un-freeze the statistics.
            set_bn_mode(model, bn_mode)
            x, y, is_forget = next(stream)
            x = x.to(device, non_blocking=True)
            is_forget = is_forget.to(device)

            with torch.no_grad():
                t_bad = incompetent(x)
                t_good = competent(x)
            # per-sample teacher selection: incompetent for forget, competent for retain
            teacher = torch.where(is_forget.view(-1, 1), t_bad, t_good)

            s_logits = model(x)
            loss = _kl(s_logits, teacher, temperature)
            if use_alpha:
                xr, _yr = next_retain()
                xr = xr.to(device, non_blocking=True)
                with torch.no_grad():
                    t_r = competent(xr)
                loss = alpha * loss + (1.0 - alpha) * _kl(model(xr), t_r, temperature)
            if opt is not None:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            else:
                sgd_descent_step(model, loss, lr)
            assert_bn_mode(model, bn_mode)

            if log and (step % 20 == 0 or step == 1):
                log(f"  [BadTeacher/{optimizer}] step {step}/{steps} "
                    f"KL={loss.item():.4f} (retain_fraction={retain_fraction})")
            history.append({"step": step, "kl": float(loss.item())})

    return model, {"history": history, "bn_mode": bn_mode,
                   "hparams": {"lr": lr, "steps": steps, "temperature": temperature,
                               "alpha": alpha, "alpha_applied": bool(use_alpha),
                               "retain_fraction": retain_fraction,
                               "optimizer": optimizer, "epochs": epochs}}
