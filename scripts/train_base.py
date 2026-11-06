#!/usr/bin/env python3
"""Train the original model on the full training set, or the retrain-on-retain reference.

    python scripts/train_base.py --which base --gpu 0
    python scripts/train_base.py --which retrain --seed 0 --gpu 0
    python scripts/train_base.py --which base --dataset cifar100 --gpu 0
    python scripts/train_base.py --which retrain --dataset cifar100 --forget-classes 0 1 2 3 4
    python scripts/train_base.py --which base --arch resnet18_gn
    python scripts/train_base.py --which base --dataset tiny_imagenet --arch resnet50
    python scripts/train_base.py --which base --dataset tiny_imagenet --arch vit_s16 \
        --epochs 10 --lr 0.01 --weight-decay 0
"""

import argparse
import os

from common import CKPT_DIR, NUM_CLASSES, ckpt_name, dump, get_device, log, make_loaders
from src.eval.accuracy import evaluate_all
from src.models.factory import ARCHS, build_model
from src.train.base import train_model
from src.utils.seeding import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["base", "retrain"], required=True)
    ap.add_argument("--dataset", choices=sorted(NUM_CLASSES), default="cifar10")
    ap.add_argument("--arch", choices=sorted(ARCHS), default="resnet18")
    ap.add_argument("--forget-class", type=int, default=0)
    ap.add_argument("--forget-classes", type=int, nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed, deterministic=True)
    device = get_device(args.gpu)

    cls_tag = (f"class{args.forget_class}" if not args.forget_classes
               else "classes" + "-".join(str(c) for c in args.forget_classes))
    tag = ckpt_name(args.which, args.dataset, args.arch, cls_tag, args.seed)[:-len(".pt")]
    os.makedirs(CKPT_DIR, exist_ok=True)

    loaders = make_loaders(
        args.dataset, mode="classes" if args.forget_classes else "class",
        forget_class=args.forget_class, forget_classes=args.forget_classes,
        seed=args.seed, batch_size=args.batch_size, unlearn_batch_size=args.batch_size,
        num_workers=args.num_workers,
        arch=args.arch)
    train_loader = loaders["train_aug"] if args.which == "base" \
        else loaders["retain_train_aug"]

    model, history = train_model(
        train_loader, loaders, device, epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, momentum=args.momentum,
        ckpt_path=os.path.join(CKPT_DIR, f"{tag}.pt"), resume=args.resume,
        config=vars(args), amp=not args.no_amp, label=args.which,
        forget_class=args.forget_class, num_classes=NUM_CLASSES[args.dataset],
        log_every=args.eval_every,
        model=build_model(args.arch, NUM_CLASSES[args.dataset],
                          pretrained=args.arch == "vit_s16"))

    final = evaluate_all(model, loaders, device)
    dump({"tag": tag, "final_metrics": final, "history": history}, f"train_{tag}.json")
    log(f"[{args.which}] {final}")


if __name__ == "__main__":
    main()
