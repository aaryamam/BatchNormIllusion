#!/usr/bin/env python3
"""Hyperparameter sweeps: GA step count, and LR/step grids for BadTeacher and SalUn.

    python scripts/run_sweeps.py --gpu 0
"""

import argparse

from common import dump, get_device, load_base, make_loaders, run_checkpoint
from src.utils.seeding import set_seed

GA_STEPS = [100, 200, 300, 400, 500, 600]
GRIDS = {"badteacher": {"lrs": [5e-4, 1e-3, 2e-3], "steps": [250, 500, 1000]},
         "salun": {"lrs": [5e-4, 1e-3, 2e-3], "steps": [100, 200, 400]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="sweeps.json")
    args = ap.parse_args()

    device = get_device(args.gpu)
    set_seed(args.seed)
    loaders = make_loaders("cifar10", forget_class=0, seed=args.seed)
    base_state = load_base("cifar10")
    out = {"ga": [], "grids": {m: [] for m in GRIDS}}

    for steps in GA_STEPS:
        row = run_checkpoint("ga", base_state, loaders, device, seed=args.seed,
                             overrides={"steps": steps},
                             tag=f"ga_steps{steps}_seed{args.seed}", subdir="sweeps")
        out["ga"].append({**row, "steps": steps})
        dump(out, args.out)

    for method, grid in GRIDS.items():
        for lr in grid["lrs"]:
            for steps in grid["steps"]:
                row = run_checkpoint(method, base_state, loaders, device, seed=args.seed,
                                     overrides={"lr": lr, "steps": steps},
                                     tag=f"{method}_lr{lr:g}_steps{steps}_seed{args.seed}",
                                     subdir="sweeps")
                out["grids"][method].append({**row, "lr": lr, "steps": steps})
                dump(out, args.out)


if __name__ == "__main__":
    main()
