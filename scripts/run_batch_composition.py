#!/usr/bin/env python3
"""Sweep the retain fraction of the unlearning loader.

    python scripts/run_batch_composition.py --gpu 0
"""

import argparse

from common import dump, get_device, load_base, make_loaders, run_checkpoint
from src.utils.seeding import set_seed

METHODS = ["badteacher", "salun"]
FRACTIONS = [0.0, 0.25, 0.5, 0.75, 0.9]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--fractions", type=float, nargs="+", default=FRACTIONS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="batch_composition.json")
    args = ap.parse_args()

    device = get_device(args.gpu)
    base_state = load_base("cifar10")
    loaders = make_loaders("cifar10", seed=args.seeds[0])
    rows = []
    for method in args.methods:
        for fraction in args.fractions:
            for seed in args.seeds:
                set_seed(seed, deterministic=True)
                row = run_checkpoint(method, base_state, loaders, device, seed=seed,
                                     overrides={"retain_fraction": fraction},
                                     tag=f"{method}_rf{fraction:g}_seed{seed}",
                                     subdir="batch_composition", recal_stream="retain_recal",
                                     do_probe=True)
                rows.append({**row, "retain_fraction": fraction})
                dump({"rows": rows}, args.out)


if __name__ == "__main__":
    main()
