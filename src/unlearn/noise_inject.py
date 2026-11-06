"""NoiseInject: a simple Gaussian noise baseline, then a retain repair. BN mode: train.

This is not the Fisher forgetting procedure of Golatkar et al.

    impair   w <- w + sigma * std(W_layer) * eps        per weight tensor, eps ~ N(0,1)
    repair   fine-tune on retain for `repair_steps` at `repair_lr`, BN in train mode

sigma is in units of each layer's weight std. With `impair_target='all'` the relative
perturbation is about 0.9 * sigma, so sigma=5 can destroy a converged network; lower it
through `overrides` as needed.
"""

import torch
import torch.nn as nn

from .common import StepBudget, assert_bn_mode, bn_mode_ctx, sgd_descent_step

BN_MODE = "train"
DEFAULTS = {"sigma_noise": 5.0, "repair_steps": 100, "repair_lr": 1e-4,
            "impair_target": "all"}


IMPAIR_TARGETS = ("all", "head", "bn_affine")


def _impair_selected(model, target, head_dim):
    """Which parameter tensors the impair step touches.

    'all'        every weight tensor (dim >= 2)
    'head'       the classifier weight matrix alone
    'bn_affine'  the BatchNorm affine parameters only
    """
    if target == "bn_affine":
        keep = set()
        for mod_name, mod in model.named_modules():
            if isinstance(mod, nn.BatchNorm2d):
                keep.add(f"{mod_name}.weight")
                keep.add(f"{mod_name}.bias")
        return [(n, p) for n, p in model.named_parameters() if n in keep]
    if target == "head":
        return [(n, p) for n, p in model.named_parameters()
                if p.dim() == 2 and p.shape[0] == head_dim]
    return [(n, p) for n, p in model.named_parameters() if p.dim() >= 2]


@torch.no_grad()
def impair(model, sigma, generator, target="all"):
    """w <- w + sigma * std(W_layer) * eps over the selected tensors."""
    if target not in IMPAIR_TARGETS:
        raise ValueError(f"impair target must be one of {IMPAIR_TARGETS}, got {target!r}")
    head_dim = model.fc.out_features
    selected = dict(_impair_selected(model, target, head_dim))
    stats, num, den = [], 0.0, 0.0
    for n, p in model.named_parameters():
        den += float(p.norm() ** 2)
        if n not in selected:
            continue
        std = float(p.detach().std())
        delta = (sigma * std) * torch.randn(p.shape, generator=generator).to(p.device)
        p.add_(delta)
        num += float(delta.norm() ** 2)
        stats.append({"param": n, "layer_std": std,
                      "rel": float(delta.norm() / (p.norm() + 1e-12))})
    return stats, (num ** 0.5) / (den ** 0.5 + 1e-12)


def unlearn_noise_inject(model, loaders, device, sigma_noise=5.0, repair_steps=100,
                          repair_lr=1e-4, impair_target="all", bn_mode=BN_MODE,
                          seed=0, log=None, **_):
    gen = torch.Generator(device="cpu").manual_seed(seed + 777)
    crit = nn.CrossEntropyLoss()
    history = []

    with bn_mode_ctx(model, bn_mode, log=log):
        stats, rel = impair(model, sigma_noise, gen, target=impair_target)
        assert_bn_mode(model, bn_mode)
        if log:
            log(f"  [NoiseInject] impair target={impair_target} "
                f"sigma={sigma_noise} on {len(stats)} tensors, "
                f"||d_theta||/||theta||={rel:.4f}")

        if repair_steps > 0:
            for step, (x, y) in StepBudget(loaders["retain_train_aug"], repair_steps):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                loss = crit(model(x), y)
                sgd_descent_step(model, loss, repair_lr)
                assert_bn_mode(model, bn_mode)
                if log and (step % 50 == 0 or step == 1):
                    log(f"  [NoiseInject] repair step {step}/{repair_steps} "
                        f"retain_CE={loss.item():.4f}")
                history.append({"step": step, "retain_ce": float(loss.item())})
        else:
            # With no repair, make one no-grad pass over retain so BN stays in train mode.
            with torch.no_grad():
                for i, (x, _y) in enumerate(loaders["retain_train_aug"]):
                    model(x.to(device, non_blocking=True))
                    if i >= 19:
                        break
            assert_bn_mode(model, bn_mode)

    return model, {"history": history, "bn_mode": bn_mode,
                   "impair_stats": stats, "rel_perturbation": rel,
                   "n_tensors_perturbed": len(stats),
                   "hparams": {"sigma_noise": sigma_noise,
                               "repair_steps": repair_steps,
                               "repair_lr": repair_lr,
                               "impair_target": impair_target}}
