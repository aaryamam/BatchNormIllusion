#!/usr/bin/env python3
"""Adversarial recovery: apply the recalibration pass with an attacker's unlabeled images.

Attacks start from the pre-recalibration checkpoints saved by scripts/run_unlearning.py.
The in-distribution attacker uses CIFAR-10 test images; the out-of-distribution attacker
uses CIFAR-100 images under CIFAR-10 normalisation.

    python scripts/run_recovery.py --gpu 0
"""

import argparse
import copy
import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

from common import (CKPT_DIR, dump, fresh_model, get_device, load_state, log, make_loaders,
                    read_results)
from src.data.cifar import CIFAR10_MEAN, CIFAR10_STD, DEFAULT_DATA_ROOT
from src.diagnostic.recalibrate import recalibrate_bn
from src.eval.accuracy import accuracy
from src.utils.seeding import loader_generator, seed_worker, set_seed

BUDGETS = [10, 25, 50, 100, 500, 1000, 2500, 5000, 9000]


def attacker_loader(dataset, n, seed, batch_size=100):
    rng = np.random.default_rng(seed + 9091)
    idx = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    return DataLoader(Subset(dataset, [int(i) for i in idx]),
                      batch_size=min(batch_size, n), shuffle=False, num_workers=2,
                      worker_init_fn=seed_worker, generator=loader_generator(seed))


def attack(model, attacker, loaders, device):
    """Recalibrate a copy on the attacker's data; return forget/retain accuracy after."""
    m = copy.deepcopy(model)
    m, _ = recalibrate_bn(m, attacker, device, inplace=True, strict=False)
    forget = accuracy(m, loaders["forget_eval_train"], device)
    retain = accuracy(m, loaders["retain_eval_train"], device)
    del m
    torch.cuda.empty_cache()
    return forget, retain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--budgets", type=int, nargs="+", default=BUDGETS)
    ap.add_argument("--ood-budget", type=int, default=100)
    ap.add_argument("--input", default="unlearning.json")
    ap.add_argument("--out", default="recovery.json")
    args = ap.parse_args()

    device = get_device(args.gpu)
    rows = [r for r in read_results(args.input)["rows"] if r["method"] != "retrain"]
    tf = T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    ood = torchvision.datasets.CIFAR100(root=DEFAULT_DATA_ROOT, train=False, transform=tf,
                                        download=True)
    sweep, ood_rows = [], []
    for r in rows:
        seed = r["seed"]
        set_seed(seed)
        loaders = make_loaders("cifar10", forget_class=0, seed=seed)
        indist = loaders["datasets"]["test_det"]
        path = os.path.join(CKPT_DIR, "table1", f"{r['tag']}_pre.pt")
        model = fresh_model(load_state(path), "cifar10", device)
        pre_f = accuracy(model, loaders["forget_eval_train"], device)

        for budget in args.budgets:
            f, ret = attack(model, attacker_loader(indist, budget, seed), loaders, device)
            sweep.append({"method": r["method"], "seed": seed, "B": budget,
                          "PreF": pre_f, "PostF": f, "PostR": ret})
            log(f"  {r['tag']:24s} B={budget:5d} F {pre_f:6.2f} -> {f:6.2f}  R={ret:6.2f}")

        auditor, _ = attack(model, loaders["retain_recal_aug"], loaders, device)
        in_f, _ = attack(model, attacker_loader(indist, args.ood_budget, seed), loaders, device)
        ood_f, _ = attack(model, attacker_loader(ood, args.ood_budget, seed), loaders, device)
        ood_rows.append({"method": r["method"], "seed": seed, "B": args.ood_budget,
                         "PreF": pre_f, "auditor": auditor, "indist": in_f, "ood": ood_f})
        dump({"sweep": sweep, "ood": ood_rows}, args.out)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
