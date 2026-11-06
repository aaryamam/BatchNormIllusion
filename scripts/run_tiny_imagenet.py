#!/usr/bin/env python3
"""ResNet-50 on Tiny-ImageNet: the method panel over three seeds, and a GA step-count sweep.

Reports Pre-F, Post-F, Pre-R, LP-50, Relearn-AUC, the breakthrough epoch and BN drift. Needs
`train_base.py --dataset tiny_imagenet --arch resnet50`, for both base and retrain (one retrain
per seed).

    python scripts/run_tiny_imagenet.py --gpu 0
"""

import argparse

from common import (dump, get_device, load_base, load_retrain, make_loaders, parse_overrides,
                    run_checkpoint)
from src.utils.seeding import set_seed

DATASET = "tiny_imagenet"
ARCH = "resnet50"
METHODS = ["scrub", "ga", "ga_ft", "salun", "incompteacher", "ssd", "noise_inject", "badteacher"]
GA_STEPS = [50, 100, 200, 400, 800]
RELEARN_LR = 3e-3
# The 3x3 stem keeps 64x64 feature maps; two gradient passes at 256 exceed a 48 GB GPU.
UNLEARN_BS = 128
# sigma = 5 destroys a converged network; see the README.
DEFAULT_OVERRIDES = {"noise_inject": {"sigma_noise": 0.5}}


def breakthrough_epoch(curve, threshold, every=10):
    """First checked relearning epoch with forget accuracy >= threshold, else the last epoch."""
    for epoch in range(0, len(curve), every):
        if curve[epoch] >= threshold:
            return epoch
    return len(curve) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--ga-steps", type=int, nargs="*", default=GA_STEPS)
    ap.add_argument("--break-threshold", type=float, default=50.0)
    ap.add_argument("--override", nargs="*", default=[], metavar="METHOD.KEY=VALUE")
    ap.add_argument("--out", default="tiny_imagenet.json")
    args = ap.parse_args()

    device = get_device(args.gpu)
    given = parse_overrides(args.override)
    base_state = load_base(DATASET, ARCH)
    common = dict(device=device, dataset=DATASET, arch=ARCH)
    out = {"methods": [], "ga_sweep": []}

    for seed in args.seeds:
        loaders = make_loaders(DATASET, seed=seed, arch=ARCH, unlearn_batch_size=UNLEARN_BS)
        jobs = [("retrain", load_retrain(DATASET, ARCH, seed))]
        jobs += [(m, base_state) for m in args.methods]
        for method, state in jobs:
            set_seed(seed, deterministic=True)
            row = run_checkpoint(method, state, loaders, seed=seed, subdir="tiny_imagenet",
                                 overrides={**DEFAULT_OVERRIDES.get(method, {}),
                                            **given.get(method, {})} or None,
                                 do_probe=True, do_relearn=True,
                                 relearn_lr=RELEARN_LR, **common)
            row["break_epoch"] = breakthrough_epoch(row["pre"]["curve"], args.break_threshold)
            row["bn_drift_sum"] = sum(row["bn_drift"])
            out["methods"].append(row)
            dump(out, args.out)

    seed = args.seeds[0]
    loaders = make_loaders(DATASET, seed=seed, arch=ARCH, unlearn_batch_size=UNLEARN_BS)
    for steps in args.ga_steps:
        set_seed(seed, deterministic=True)
        row = run_checkpoint("ga", base_state, loaders, seed=seed, overrides={"steps": steps},
                             tag=f"ga_steps{steps}_seed{seed}", subdir="tiny_imagenet", **common)
        row.update(steps=steps, bn_drift_sum=sum(row["bn_drift"]))
        out["ga_sweep"].append(row)
        dump(out, args.out)


if __name__ == "__main__":
    main()
