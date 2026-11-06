#!/usr/bin/env python3
"""Instance-wise forgetting: forget samples drawn at random across all classes.

Three seeds at p = 0.10 and one seed at the other fractions. With no forget class, LP-50 and
NCC report overall test accuracy.

    python scripts/run_instance_wise.py --gpu 0
"""

import argparse

from common import dump, get_device, load_base, make_loaders, run_checkpoint
from src.utils.seeding import set_seed

METHODS = ["salun", "badteacher", "scrub", "ga"]
FRACTIONS = [0.01, 0.05, 0.10]
MAIN_FRACTION = 0.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--fractions", type=float, nargs="+", default=FRACTIONS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="instance_wise.json")
    args = ap.parse_args()

    device = get_device(args.gpu)
    base_state = load_base("cifar10")
    rows = []
    for p in args.fractions:
        for seed in args.seeds if p == MAIN_FRACTION else args.seeds[:1]:
            loaders = make_loaders("cifar10", mode="iid_random", p=p, seed=seed)
            for method in args.methods:
                set_seed(seed, deterministic=True)
                row = run_checkpoint(method, base_state, loaders, device, seed=seed,
                                     tag=f"{method}_p{p:g}_seed{seed}", subdir="instance_wise",
                                     recal_stream="retain_recal", do_probe=True,
                                     forget_class=None)
                rows.append({**row, "p": p,
                             "LP_pre": row["pre"]["lp50_overall_acc"],
                             "LP_post": row["post"]["lp50_overall_acc"],
                             "NCC_pre": row["pre"]["ncc_overall_acc"],
                             "NCC_post": row["post"]["ncc_overall_acc"]})
                dump({"rows": rows}, args.out)


if __name__ == "__main__":
    main()
