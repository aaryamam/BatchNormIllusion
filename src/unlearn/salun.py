"""SalUn: saliency-masked unlearning (Fan et al.). BN mode: train.

`objective='random_labeling'` (default) descends on randomly relabelled forget samples
under the saliency mask. `objective='grad_ascent'` is masked gradient ascent.

`threshold=0.5` keeps the top 50% of parameters by |grad|, measured on the forget set.
"""

import torch
import torch.nn as nn

from ..data.cifar import make_mixed_unlearn_loader
from .common import StepBudget, assert_bn_mode, bn_mode_ctx, head_size
from .random_labeling import random_wrong_labels

BN_MODE = "train"
DEFAULTS = {"lr": 1e-3, "steps": 200, "threshold": 0.5,
            "objective": "random_labeling", "retain_fraction": 0.0}


def compute_saliency_mask(model, forget_loader, device, threshold=0.5,
                          saliency_batches=None, bn_mode=BN_MODE):
    """Binary per-parameter mask keeping the top (1 - threshold) fraction by |grad|.

    The saliency pass accumulates the gradient of the forget-set CE loss. It is run
    under the method's declared BN mode so the gradients are the ones the unlearning
    loop will actually see.
    """
    crit = nn.CrossEntropyLoss()
    model.zero_grad(set_to_none=True)
    accum = {n: torch.zeros_like(p) for n, p in model.named_parameters()}

    prev = model.training
    from .common import set_bn_mode
    set_bn_mode(model, bn_mode)

    n_batches = 0
    for i, (x, y) in enumerate(forget_loader):
        if saliency_batches is not None and i >= saliency_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        model.zero_grad(set_to_none=True)
        crit(model(x), y).backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                accum[n] += p.grad.detach().abs()
        n_batches += 1
    model.zero_grad(set_to_none=True)
    model.train(prev)

    flat = torch.cat([v.flatten() for v in accum.values()])
    cutoff = torch.quantile(flat.float(), threshold) if flat.numel() < 16_000_000 \
        else torch.kthvalue(flat.float().cpu(),
                            max(1, int(threshold * flat.numel()))).values.to(flat.device)
    mask = {n: (v >= cutoff).to(v.dtype) for n, v in accum.items()}

    kept = int(sum(int(m.sum().item()) for m in mask.values()))
    total = int(sum(m.numel() for m in mask.values()))
    return mask, {"saliency_batches": n_batches, "threshold": threshold,
                  "cutoff": float(cutoff), "params_kept": kept, "params_total": total,
                  "kept_fraction": kept / max(total, 1)}


def unlearn_salun(model, loaders, device, lr=1e-3, steps=200, threshold=0.5,
                  objective="random_labeling", retain_fraction=0.0,
                  saliency_batches=None, bn_mode=BN_MODE, seed=0, log=None, **_):
    """`retain_fraction` > 0 uses a mixed forget/retain unlearning loader.

    Forget samples get a uniformly random wrong label, retain samples keep their true label,
    and both are descended on under the saliency mask.
    """
    if objective not in ("random_labeling", "grad_ascent"):
        raise ValueError(f"objective must be 'random_labeling' or 'grad_ascent', "
                         f"got {objective!r}")

    mask, mask_info = compute_saliency_mask(
        model, loaders["forget_train_aug"], device, threshold=threshold,
        saliency_batches=saliency_batches, bn_mode=bn_mode)
    if log:
        log(f"  [SalUn/{objective}] saliency mask keeps "
            f"{mask_info['kept_fraction']:.1%} of {mask_info['params_total']:,} params "
            f"(threshold={threshold})")

    crit = nn.CrossEntropyLoss()
    torch.manual_seed(seed + 4242)
    # Follow the student's head so this works for any number of classes.
    n_classes = head_size(model)

    mixed = retain_fraction > 0.0
    if mixed:
        bs = loaders["config"]["batch_size"]
        stream = iter(make_mixed_unlearn_loader(loaders, retain_fraction, bs))
    else:
        stream = ((x, y, torch.ones(x.size(0), dtype=torch.bool))
                  for _s, (x, y) in StepBudget(loaders["forget_train_aug"], steps))

    history = []
    with bn_mode_ctx(model, bn_mode, log=log):
        for step in range(1, steps + 1):
            x, y, is_forget = next(stream)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            is_forget = is_forget.to(device)

            if objective == "random_labeling":
                # forget samples -> random wrong label; retain samples keep the true label
                target = torch.where(is_forget,
                                     random_wrong_labels(y, n_classes), y)
                sign = -1.0            # DESCEND onto the (re)labelled targets
            else:
                target = y
                sign = +1.0            # ASCEND away from the true labels

            loss = crit(model(x), target)
            model.zero_grad(set_to_none=True)
            loss.backward()
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if p.grad is not None:
                        p.data.add_(p.grad * mask[n], alpha=sign * lr)
            assert_bn_mode(model, bn_mode)
            if log and (step % 50 == 0 or step == 1):
                log(f"  [SalUn/{objective}] step {step}/{steps} loss={loss.item():.4f} "
                    f"(retain_fraction={retain_fraction})")
            history.append({"step": step, "loss": float(loss.item())})

    return model, {"history": history, "bn_mode": bn_mode, "mask": mask_info,
                   "hparams": {"lr": lr, "steps": steps, "threshold": threshold,
                               "objective": objective,
                               "retain_fraction": retain_fraction}}
