#!/usr/bin/env python3
"""CIFAR-100: single-class (class 0) and five-class (classes 0-4) forgetting.

Reports forget accuracy, LP-50, NCC and Relearn-AUC before and after recalibration. The
five-class relearning attacker gets 50 samples per forget class.

    python scripts/run_cifar100.py --gpu 0
"""

import argparse
import copy
import os

import torch

from common import (CKPT_DIR, dump, fresh_model, get_device, load_base, load_state, log,
                    make_loaders, measure, save_pair, summarize)
from src.diagnostic.recalibrate import recalibrate_bn
from src.eval.relearn import relearn_auc
from src.unlearn.registry import run_method
from src.utils.seeding import set_seed

RELEARN_LR = 3e-4
CONDITIONS = {
    "single_class": {"mode": "class", "forget_classes": None,
                     "methods": ["scrub", "ga", "ga_ft", "salun", "badteacher",
                                 "incompteacher", "noise_inject", "ssd"],
                     "retrain": "cifar100_retrain_class0_seed0.pt",
                     "relearn": {"n_samples": 50, "per_class": None}},
    "five_class": {"mode": "classes", "forget_classes": [0, 1, 2, 3, 4],
                   "methods": ["scrub", "ga", "salun"],
                   "retrain": "cifar100_retrain_classes0-1-2-3-4_seed0.pt",
                   "relearn": {"n_samples": 250, "per_class": 50}},
}


def measure_pair(model, loaders, device, seed, relearn, forget):
    """Pre -> recalibrate -> post, with LP-50, NCC and Relearn-AUC on both states."""
    def panel(m):
        out = measure(m, loaders, device, forget, 100, seed, do_probe=True)
        out.update(relearn_auc(m, loaders, device, seed=seed, lr=RELEARN_LR, **relearn))
        return out

    pre = panel(model)
    pre_state = copy.deepcopy(model.state_dict())
    model, _ = recalibrate_bn(model, loaders["retain_recal_aug"], device, inplace=True)
    return pre, panel(model), pre_state, model


def row_of(method, seed, tag, pre, post):
    return summarize(method, seed, tag, pre, post,
                     LP_pre=pre["lp50_forget_acc"], LP_post=post["lp50_forget_acc"],
                     NCC_pre=pre["ncc_forget_acc"], NCC_post=post["ncc_forget_acc"],
                     ReAUC_pre=pre["relearn_auc"], ReAUC_post=post["relearn_auc"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--out", default="cifar100.json")
    args = ap.parse_args()

    device = get_device(args.gpu)
    base_state = load_base("cifar100")
    out = {}
    for name in args.conditions:
        cfg = CONDITIONS[name]
        set_seed(args.seed)
        loaders = make_loaders("cifar100", forget_class=0, seed=args.seed, mode=cfg["mode"],
                               forget_classes=cfg["forget_classes"])
        out[name] = []

        states = [("retrain", load_state(os.path.join(CKPT_DIR, cfg["retrain"])))]
        states += [(m, base_state) for m in cfg["methods"]]
        for method, state in states:
            log(f"\n{name} / {method}")
            model = fresh_model(state, "cifar100", device)
            if method != "retrain":
                model, _ = run_method(method, model, loaders, device, seed=args.seed,
                                      log=log, inplace=True)
            pre, post, pre_state, model = measure_pair(model, loaders, device, args.seed,
                                                       cfg["relearn"],
                                                       cfg["forget_classes"] or 0)
            tag = f"{name}_{method}"
            save_pair(pre_state, model.state_dict(), "cifar100", tag)
            out[name].append(row_of(method, args.seed, tag, pre, post))
            del model
            torch.cuda.empty_cache()
            dump(out, args.out)


if __name__ == "__main__":
    main()
