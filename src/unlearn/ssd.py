"""SSD: Selective Synaptic Dampening (Foster et al., AAAI 2024). BN mode: train.

Reimplementation of github.com/if-loops/selective-synaptic-dampening:

  Selection   locations = { i : fimp_i  >  alpha * oimp_i }
  Dampening   p_i <- p_i * min( 1, ( lambda * oimp_i / fimp_i ) ** exponent )

`oimp` and `fimp` are diagonal-Fisher importances over the full and forget sets.

Unlike the original, which calls `model.eval()`, the importance passes run with BN in train
mode here, so they update the running statistics.
"""

import torch
import torch.nn as nn

from .common import assert_bn_mode, bn_mode_ctx

BN_MODE = "train"
DEFAULTS = {"dampening_constant": 10000.0, "selection_weighting": 10.0,
            "exponent": 1.0, "lower_bound": 1.0}


def _calc_importance(model, loader, device, bn_mode, max_batches=None):
    """Mean-over-batches diagonal Fisher."""
    crit = nn.CrossEntropyLoss()
    imp = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    n_batches = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        model.zero_grad(set_to_none=True)
        crit(model(x), y).backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                imp[n] += p.grad.detach().clone().pow(2)
        n_batches += 1
        assert_bn_mode(model, bn_mode)
    model.zero_grad(set_to_none=True)
    for n in imp:
        imp[n] /= float(max(n_batches, 1))
    return imp, n_batches


def unlearn_ssd(model, loaders, device, dampening_constant=10000.0,
                selection_weighting=10.0, exponent=1.0, lower_bound=1.0,
                bn_mode=BN_MODE, seed=0, log=None, **_):
    history = []
    with bn_mode_ctx(model, bn_mode, log=log):
        fimp, n_f = _calc_importance(model, loaders["forget_train_aug"], device, bn_mode)
        if log:
            log(f"  [SSD] forget importance over {n_f} batches")
        oimp, n_o = _calc_importance(model, loaders["train_aug"], device, bn_mode)
        if log:
            log(f"  [SSD] full-data importance over {n_o} batches")

        n_selected, n_dampened, n_total = 0, 0, 0
        with torch.no_grad():
            for n, p in model.named_parameters():
                o, f = oimp[n], fimp[n]
                locations = torch.where(f > o.mul(selection_weighting))
                weight = (o.mul(dampening_constant).div(f + 1e-12)).pow(exponent)
                update = weight[locations]
                update[torch.where(update > lower_bound)] = lower_bound
                p[locations] = p[locations].mul(update)

                n_total += p.numel()
                n_selected += int(locations[0].numel())
                n_dampened += int((update < lower_bound).sum().item())
                history.append({"param": n, "selected": int(locations[0].numel()),
                                "dampened": int((update < lower_bound).sum().item()),
                                "numel": int(p.numel())})
        assert_bn_mode(model, bn_mode)

    if log:
        log(f"  [SSD] selected {n_selected:,}/{n_total:,} params "
            f"({100*n_selected/max(n_total,1):.3f}%), actually dampened {n_dampened:,}")

    return model, {"history": history, "bn_mode": bn_mode,
                   "n_params_selected": n_selected, "n_params_dampened": n_dampened,
                   "n_params_total": n_total,
                   "hparams": {"dampening_constant": dampening_constant,
                               "selection_weighting": selection_weighting,
                               "exponent": exponent, "lower_bound": lower_bound}}
