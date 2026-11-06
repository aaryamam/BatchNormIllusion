#!/usr/bin/env python3
"""LayerNorm control: ViT-S/16 (ImageNet-pretrained, fine-tuned on Tiny-ImageNet).

The model has no running statistics, so the pass must leave every tensor bit-identical and
forget accuracy unchanged. Needs `train_base.py --dataset tiny_imagenet --arch vit_s16`, for
both base and retrain. The settings below are starting points: pass `--override` until each
method reaches a forgetting regime.

    python scripts/run_vit.py --gpu 0 --override ga.lr=1e-4 ga.steps=200
"""

import argparse

from common import (dump, get_device, load_base, load_retrain, make_loaders, parse_overrides,
                    run_checkpoint)
from src.utils.seeding import set_seed

DATASET = "tiny_imagenet"
ARCH = "vit_s16"
DEFAULT_OVERRIDES = {m: {"lr": 1e-4, "steps": 100} for m in ["ga", "salun", "badteacher", "scrub"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", nargs="+", default=list(DEFAULT_OVERRIDES))
    ap.add_argument("--override", nargs="*", default=[], metavar="METHOD.KEY=VALUE")
    ap.add_argument("--out", default="vit.json")
    args = ap.parse_args()

    device = get_device(args.gpu)
    given = parse_overrides(args.override)
    base_state = load_base(DATASET, ARCH)
    loaders = make_loaders(DATASET, seed=args.seed, arch=ARCH)
    jobs = [("retrain", load_retrain(DATASET, ARCH, args.seed))]
    jobs += [(m, base_state) for m in args.methods]
    rows = []
    for method, state in jobs:
        set_seed(args.seed, deterministic=True)
        hp = {**DEFAULT_OVERRIDES.get(method, {}), **given.get(method, {})} or None
        rows.append(run_checkpoint(method, state, loaders, device, seed=args.seed, dataset=DATASET,
                                   arch=ARCH, overrides=hp, subdir="vit"))
        dump({"rows": rows}, args.out)


if __name__ == "__main__":
    main()
