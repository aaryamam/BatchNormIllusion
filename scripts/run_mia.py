#!/usr/bin/env python3
"""Membership-inference attacks before and after recalibration.

Members are forget-class train samples; non-members are forget-class test samples.

    python scripts/run_mia.py --gpu 0 --seeds 0 1 2
"""

import argparse

import torch

from common import dump, fresh_model, get_device, load_base, log, make_loaders
from src.diagnostic.recalibrate import recalibrate_bn
from src.eval.accuracy import accuracy
from src.eval.mia import ATTACKS, run_mia
from src.unlearn.registry import run_method
from src.utils.seeding import set_seed

METHODS = ["salun", "badteacher", "scrub", "ga"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--out", default="mia.json")
    args = ap.parse_args()

    device = get_device(args.gpu)
    base_state = load_base("cifar10")
    rows = []
    for seed in args.seeds:
        set_seed(seed, deterministic=True)
        loaders = make_loaders("cifar10", forget_class=0, seed=seed)
        for method in args.methods:
            model = fresh_model(base_state, "cifar10", device)
            model, _ = run_method(method, model, loaders, device, seed=seed, log=log,
                                  inplace=True)
            pre_f = accuracy(model, loaders["forget_eval_train"], device)
            mia_pre = run_mia(model, loaders, device)
            model, _ = recalibrate_bn(model, loaders["retain_recal"], device, inplace=True)
            post_f = accuracy(model, loaders["forget_eval_train"], device)
            mia_post = run_mia(model, loaders, device)
            rows.append({"method": method, "seed": seed, "PreF": pre_f, "PostF": post_f,
                         "mia_pre": mia_pre, "mia_post": mia_post})
            log(f"  {method} seed {seed}: dF={post_f - pre_f:+.1f} " + " ".join(
                f"{a}: {mia_pre[a]['auc']:.3f}->{mia_post[a]['auc']:.3f}" for a in ATTACKS))
            del model
            torch.cuda.empty_cache()
            dump({"rows": rows}, args.out)


if __name__ == "__main__":
    main()
