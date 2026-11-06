"""The diagnostic: BN recalibration on retain data, plus per-layer drift measurement.

The operator is `torch.optim.swa_utils.update_bn(retain_loader, model)`. It runs the model
in train mode with gradients disabled, resets every BN module's running statistics, and
uses a cumulative moving average (momentum=None), so the result does not depend on the
incoming buffer values. No learnable parameter changes, which `recalibrate_bn` records.
"""

import copy

import torch
from torch.optim.swa_utils import update_bn

from ..models.resnet_cifar import bn_layers, bn_state

MIN_RECAL_SAMPLES = 5000


def _count_samples(loader):
    ds = getattr(loader, "dataset", None)
    if ds is not None:
        try:
            return len(ds)
        except TypeError:
            pass
    return -1


def _forward_pass_is_identity(model, loader, device):
    """The same pass on a model with no running statistics: nothing may change."""
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    was_training = model.training
    model.train()
    with torch.no_grad():
        for x, *_ in loader:
            model(x.to(device, non_blocking=True))
    model.train(was_training)
    return all(torch.equal(before[k], v.detach().cpu())
               for k, v in model.state_dict().items())


def recalibrate_bn(model, retain_loader, device, min_samples=MIN_RECAL_SAMPLES,
                   inplace=True, strict=True):
    """Apply R: reset and recompute BN running statistics from retain data.

    Returns (model, info). `info` records the sample count, whether the weights stayed
    bit-identical, and per-layer drift.
    """
    n = _count_samples(retain_loader)
    if strict and 0 <= n < min_samples:
        raise ValueError(
            f"recalibration needs >= {min_samples} retain samples, loader has {n}")

    if not inplace:
        model = copy.deepcopy(model)

    bns = bn_layers(model)
    if not bns:
        return model, {"n_bn_layers": 0, "note": "no BatchNorm layers",
                       "weights_bit_identical": _forward_pass_is_identity(
                           model, retain_loader, device)}

    before = bn_state(model)
    params_before = {k: v.detach().cpu().clone()
                     for k, v in model.named_parameters()}

    was_training = model.training
    # update_bn sets train mode and no_grad internally.
    update_bn(retain_loader, model, device=device)
    model.train(was_training)

    after = bn_state(model)

    weights_bit_identical = all(
        torch.equal(params_before[k], v.detach().cpu())
        for k, v in model.named_parameters()
    )

    info = {
        "n_recal_samples": n,
        "n_bn_layers": len(bns),
        "weights_bit_identical": bool(weights_bit_identical),
        "num_batches_tracked": [int(m.num_batches_tracked.item()) for _, m in bns],
        "drift": bn_drift(before, after),
    }
    return model, info


def bn_drift(state_a, state_b, eps=1e-8):
    """Per-layer normalised drift between two BN buffer snapshots.

    For each layer l:
        delta_mu_l  = || mu_a - mu_b ||_2 / || sqrt(var_b) ||_2
        delta_var_l = || var_a - var_b ||_2 / || var_b ||_2
    plus per-channel max, so a single blown-up channel is visible rather than averaged
    away. `state_b` is the reference (the recalibrated / retain-optimal statistics).
    """
    out = {}
    for i, name in enumerate(state_a.keys()):
        mu_a, var_a = state_a[name]
        mu_b, var_b = state_b[name]
        sigma_b = torch.sqrt(var_b.clamp_min(0) + eps)
        dmu = mu_a - mu_b
        dvar = var_a - var_b
        out[name] = {
            "index": i,
            "channels": int(mu_a.numel()),
            "delta_mu_l2_over_sigma": float(dmu.norm() / (sigma_b.norm() + eps)),
            "delta_mu_max_per_channel": float((dmu.abs() / (sigma_b + eps)).max()),
            "delta_mu_mean_per_channel": float((dmu.abs() / (sigma_b + eps)).mean()),
            "delta_var_l2_rel": float(dvar.norm() / (var_b.norm() + eps)),
            "delta_var_max_per_channel": float((dvar.abs() / (var_b.abs() + eps)).max()),
            "log_var_ratio_mean": float(
                (torch.log(var_a.clamp_min(eps)) - torch.log(var_b.clamp_min(eps))).mean()),
        }
    return out


def bn_buffer_maxdiff(state_a, state_b):
    """max |phi_a - phi_b| over every BN buffer element."""
    worst_mu, worst_var = 0.0, 0.0
    for name in state_a:
        mu_a, var_a = state_a[name]
        mu_b, var_b = state_b[name]
        worst_mu = max(worst_mu, float((mu_a - mu_b).abs().max()))
        worst_var = max(worst_var, float((var_a - var_b).abs().max()))
    return {"max_abs_diff_running_mean": worst_mu,
            "max_abs_diff_running_var": worst_var,
            "max_abs_diff_overall": max(worst_mu, worst_var)}


def weights_equal(model_a, model_b):
    """True iff every parameter tensor is bit-identical (torch.equal)."""
    sa = dict(model_a.named_parameters())
    sb = dict(model_b.named_parameters())
    if sa.keys() != sb.keys():
        return False
    return all(torch.equal(sa[k].detach().cpu(), sb[k].detach().cpu()) for k in sa)
