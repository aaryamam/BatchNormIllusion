#!/usr/bin/env python3
"""GroupNorm control: the same methods on a ResNet-18 with no running statistics.

Needs `train_base.py --arch resnet18_gn`, for both base and retrain.

    python scripts/run_groupnorm.py --gpu 0
"""

import argparse

from common import (dump, get_device, load_base, load_retrain, make_loaders, parse_overrides,
                    run_checkpoint)
from src.utils.seeding import set_seed

ARCH = "resnet18_gn"
METHODS = ["ga", "salun", "badteacher", "incompteacher", "scrub", "ga_ft", "noise_inject", "ssd"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--override", nargs="*", default=[], metavar="METHOD.KEY=VALUE")
    ap.add_argument("--out", default="groupnorm.json")
    args = ap.parse_args()

    device = get_device(args.gpu)
    overrides = parse_overrides(args.override)
    base_state = load_base("cifar10", ARCH)
    loaders = make_loaders("cifar10", seed=args.seed)
    jobs = [("retrain", load_retrain("cifar10", ARCH, args.seed))]
    jobs += [(m, base_state) for m in args.methods]
    rows = []
    for method, state in jobs:
        set_seed(args.seed, deterministic=True)
        rows.append(run_checkpoint(method, state, loaders, device, seed=args.seed, arch=ARCH,
                                   overrides=overrides.get(method), subdir="groupnorm",
                                   do_probe=True, do_relearn=True))
        dump({"rows": rows}, args.out)


if __name__ == "__main__":
    main()
