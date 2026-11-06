#!/usr/bin/env python3
"""Layerwise normalization mismatch, batch-size sensitivity and idempotence of recalibration.

Depth: each unlearned checkpoint against its own retain-recalibrated copy (N_r = 5000, b = 128,
fixed subset and batch order), for all 20 BN layers, plotted to results/bn_depth.pdf. Batch
size: recalibrate at each b with N_r fixed. Idempotence: apply the pass twice on identical
batches, and again with the batches reshuffled.

    python scripts/run_bn_analysis.py --gpu 0
"""

import argparse
import copy
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from common import RESULT_DIR, dump, fresh_model, get_device, load_base, log, make_loaders
from src.diagnostic.recalibrate import (bn_buffer_maxdiff, bn_drift, recalibrate_bn,
                                        weights_equal)
from src.eval.accuracy import accuracy
from src.models.resnet_cifar import bn_state
from src.unlearn.registry import run_method
from src.utils.seeding import loader_generator, seed_worker, set_seed

METHODS = ["ga", "salun", "scrub", "badteacher"]
BATCH_SIZES = [8, 16, 32, 64, 128, 256, 512]
N_RECAL = 5000


def recal_loader(loaders, batch_size, n_samples, seed):
    """A fixed retain subset of exactly `n_samples`, in fixed order."""
    r_idx = loaders["idx"]["retain_train"]
    take = np.random.default_rng(seed).choice(len(r_idx), size=n_samples, replace=False)
    subset = Subset(loaders["datasets"]["train_det"], [int(r_idx[i]) for i in sorted(take)])
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2,
                      worker_init_fn=seed_worker, generator=loader_generator(seed))


def depth_profile(model, loaders, device, args):
    ref, _ = recalibrate_bn(copy.deepcopy(model),
                            recal_loader(loaders, 128, args.n_recal, args.seed), device)
    drift = bn_drift(bn_state(model), bn_state(ref))
    return [{"name": n, **d} for n, d in drift.items()]


def batch_size_sweep(model, loaders, device, args):
    out, ref = {}, None
    for b in args.batch_sizes:
        m, _ = recalibrate_bn(copy.deepcopy(model),
                              recal_loader(loaders, b, args.n_recal, args.seed), device)
        state = bn_state(m)
        ref = ref or state
        deltas = [d["delta_mu_l2_over_sigma"] for d in bn_drift(state, ref).values()]
        out[b] = {"forget_train": accuracy(m, loaders["forget_eval_train"], device),
                  "retain_train": accuracy(m, loaders["retain_eval_train"], device),
                  "mean_delta_mu_vs_smallest_b": float(np.mean(deltas)),
                  "per_layer_delta_mu_vs_smallest_b": deltas}
        log(f"  b={b:4d} F={out[b]['forget_train']:6.2f} R={out[b]['retain_train']:6.2f}")
        del m
        torch.cuda.empty_cache()
    return out


def idempotence(model, loaders, device, args):
    loader = recal_loader(loaders, 128, args.n_recal, args.seed)
    m, _ = recalibrate_bn(copy.deepcopy(model), loader, device)
    once, weights = bn_state(m), copy.deepcopy(m)
    m, _ = recalibrate_bn(m, loader, device)
    same_batches = bn_buffer_maxdiff(once, bn_state(m))
    m, _ = recalibrate_bn(m, recal_loader(loaders, 128, args.n_recal, args.seed + 991), device)
    return {"identical_batches": same_batches,
            "reshuffled_batches": bn_buffer_maxdiff(once, bn_state(m)),
            "weights_bit_identical": weights_equal(weights, m)}


def plot_depth(depth):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for method, layers in depth.items():
        axes[0].plot([d["delta_mu_l2_over_sigma"] for d in layers], marker="o", ms=3, label=method)
        axes[1].plot([d["delta_var_l2_rel"] for d in layers], marker="o", ms=3, label=method)
    axes[0].set_ylabel("Normalized running-mean mismatch")
    axes[1].set_ylabel("Relative running-variance mismatch")
    for ax in axes:
        ax.set_xlabel("BN layer (module order)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    path = os.path.join(RESULT_DIR, "bn_depth.pdf")
    fig.savefig(path)
    log(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=BATCH_SIZES)
    ap.add_argument("--n-recal", type=int, default=N_RECAL)
    ap.add_argument("--out", default="bn_analysis.json")
    args = ap.parse_args()

    device = get_device(args.gpu)
    base_state = load_base("cifar10")
    loaders = make_loaders("cifar10", seed=args.seed)
    out = {"depth": {}, "batch_size": {}, "idempotence": {}}
    for method in args.methods:
        log(f"\n{method}")
        set_seed(args.seed, deterministic=True)
        model = fresh_model(base_state, "cifar10", device)
        model, _ = run_method(method, model, loaders, device, seed=args.seed, log=log, inplace=True)
        out["depth"][method] = depth_profile(model, loaders, device, args)
        out["batch_size"][method] = batch_size_sweep(model, loaders, device, args)
        out["idempotence"][method] = idempotence(model, loaders, device, args)
        del model
        torch.cuda.empty_cache()
        dump(out, args.out)
    plot_depth(out["depth"])


if __name__ == "__main__":
    main()
