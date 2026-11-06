#!/usr/bin/env python3
"""Run unlearning methods on CIFAR-10 (class 0) and measure the recalibration gap.

Writes results/unlearning.json and saves each pre- and post-recalibration checkpoint under
checkpoints/table1/. The other scripts read those checkpoints back.

    python scripts/run_unlearning.py --gpu 0 --seeds 0 1 2
    python scripts/run_unlearning.py --methods salun --override steps=100
"""

import argparse
import ast
import os

import torch

from common import (CKPT_DIR, dump, fresh_model, get_device, load_base, load_state, log,
                    make_loaders, measure, run_checkpoint, save_pair, summarize)
from src.diagnostic.recalibrate import recalibrate_bn
from src.utils.seeding import set_seed

METHODS = ["scrub", "ga", "ga_ft", "salun", "badteacher", "incompteacher",
           "noise_inject", "ssd"]


def retrain_row(seed, loaders, device):
    """The Retrain reference, measured with the same pre/post protocol."""
    path = os.path.join(CKPT_DIR, f"retrain_class0_seed{seed}.pt")
    if not os.path.isfile(path):
        log(f"skipping Retrain for seed {seed}: {path} not found")
        return None
    state = load_state(path)
    model = fresh_model(state, "cifar10", device)
    pre = measure(model, loaders, device)
    model, _ = recalibrate_bn(model, loaders["retain_recal_aug"], device, inplace=True)
    post = measure(model, loaders, device)
    tag = f"retrain_seed{seed}"
    save_pair(state, model.state_dict(), "table1", tag)
    del model
    torch.cuda.empty_cache()
    return summarize("retrain", seed, tag, pre, post, bn_mode="n/a")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                    help="hyperparameter overrides applied to every selected method")
    ap.add_argument("--out", default="unlearning.json")
    args = ap.parse_args()

    device = get_device(args.gpu)
    base_state = load_base("cifar10")
    overrides = {k: ast.literal_eval(v)
                 for k, v in (o.split("=", 1) for o in args.override)} or None

    rows = []
    for seed in args.seeds:
        set_seed(seed)
        loaders = make_loaders("cifar10", forget_class=0, seed=seed)
        row = retrain_row(seed, loaders, device)
        if row:
            rows.append(row)
        for method in args.methods:
            log(f"\n{method} seed {seed}")
            rows.append(run_checkpoint(method, base_state, loaders, device, seed=seed,
                                       overrides=overrides))
            dump({"rows": rows}, args.out)
    dump({"rows": rows}, args.out)


if __name__ == "__main__":
    main()
