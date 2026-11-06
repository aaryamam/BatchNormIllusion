#!/usr/bin/env python3
"""RandomLabeling, a purely data-based method, with BN in train mode and in eval mode.

    python scripts/run_random_labeling.py --gpu 0
"""

import argparse

from common import dump, get_device, load_base, make_loaders, run_checkpoint
from src.utils.seeding import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="random_labeling.json")
    args = ap.parse_args()

    device = get_device(args.gpu)
    base_state = load_base("cifar10")
    loaders = make_loaders("cifar10", seed=args.seeds[0])
    rows = []
    for bn_mode in ("train", "eval"):
        for seed in args.seeds:
            set_seed(seed, deterministic=True)
            rows.append(run_checkpoint("random_labeling", base_state, loaders, device, seed=seed,
                                       overrides={"steps": args.steps, "lr": args.lr},
                                       tag=f"random_labeling_bn{bn_mode}_seed{seed}",
                                       subdir="random_labeling", recal_stream="retain_recal",
                                       bn_mode=bn_mode, do_probe=True, do_relearn=True))
            dump({"rows": rows}, args.out)


if __name__ == "__main__":
    main()
