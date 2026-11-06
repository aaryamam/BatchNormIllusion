#!/usr/bin/env python3
"""Class-conditional activation divergence in the pre-unlearning model, at every BN layer.

For each BN layer, compare the per-channel moments of its input over forget and retain data:
delta_l = ||mu_f - mu_r||_2 / ||sigma_l||_2, with sigma_l the layer's own running std.

    python scripts/run_divergence.py --gpu 0
"""

import argparse

import numpy as np
import torch

from common import dump, fresh_model, get_device, load_base, log, make_loaders
from src.models.resnet_cifar import bn_layers
from src.utils.seeding import set_seed

FORGET_CLASSES = [0, 3, 8]  # airplane, cat, ship


@torch.no_grad()
def input_moments(model, loader, device):
    """Exact per-channel mean and variance of each BN layer's input over `loader`."""
    totals = {}

    def capture(name):
        def hook(_mod, inp, _out):
            x = inp[0].detach().float()
            s, ss = x.sum(dim=(0, 2, 3)).cpu(), (x * x).sum(dim=(0, 2, 3)).cpu()
            n = x.shape[0] * x.shape[2] * x.shape[3]
            t = totals.setdefault(name, [0.0, 0.0, 0])
            t[0], t[1], t[2] = t[0] + s, t[1] + ss, t[2] + n
        return hook

    hooks = [m.register_forward_hook(capture(n)) for n, m in bn_layers(model)]
    model.eval()
    for x, _y in loader:
        model(x.to(device, non_blocking=True))
    for h in hooks:
        h.remove()
    moments = {}
    for name, (s, ss, n) in totals.items():
        mu = s / n
        moments[name] = (mu, (ss / n - mu * mu).clamp_min(0))
    return moments


def divergence(model, forget, retain, eps=1e-8):
    out = []
    for name, m in bn_layers(model):
        mu_f, var_f = forget[name]
        mu_r, var_r = retain[name]
        sigma = torch.sqrt(m.running_var.detach().cpu().clamp_min(0) + eps)
        out.append({"name": name,
                    "delta": float((mu_f - mu_r).norm() / (sigma.norm() + eps)),
                    "var_ratio_mean": float((var_f.clamp_min(eps) / var_r.clamp_min(eps)).mean())})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--forget-classes", type=int, nargs="+", default=FORGET_CLASSES)
    ap.add_argument("--out", default="divergence.json")
    args = ap.parse_args()

    set_seed(args.seed, deterministic=True)
    device = get_device(args.gpu)
    model = fresh_model(load_base("cifar10"), "cifar10", device)
    out = {}
    for fc in args.forget_classes:
        loaders = make_loaders("cifar10", forget_class=fc, seed=args.seed)
        layers = divergence(model, input_moments(model, loaders["forget_eval_train"], device),
                            input_moments(model, loaders["retain_eval_train"], device))
        vals = [d["delta"] for d in layers]
        out[str(fc)] = {"layers": layers, "min": float(np.min(vals)), "max": float(np.max(vals)),
                        "early_mean": float(np.mean(vals[:7])), "late_mean": float(np.mean(vals[14:]))}
        log(f"class {fc}: min={out[str(fc)]['min']:.3f} max={out[str(fc)]['max']:.3f}")
        dump(out, args.out)


if __name__ == "__main__":
    main()
