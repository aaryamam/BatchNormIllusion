#!/usr/bin/env python3
"""ImageNet-100 at 224: BadTeacher on an ImageNet-pretrained ResNet-50, single-class forgetting.

Reads the Hugging Face dataset `clane9/imagenet-100`. Labels map into the pretrained model's
1000-way index space, so its head is used unmodified. Measures forget and retain accuracy
before and after full-retain recalibration, and after recalibrating on ten images.

    python scripts/run_imagenet100.py --gpu 0
"""

import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import ResNet50_Weights, resnet50

from common import dump, get_device, log
from src.diagnostic.recalibrate import recalibrate_bn
from src.eval.accuracy import accuracy
from src.unlearn.common import assert_bn_mode, bn_mode_ctx
from src.utils.seeding import loader_generator, seed_worker, set_seed

CLASS_MAP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "src", "data", "imagenet100_class_map.json")
UNLEARN_BS = 256
RECAL_BS = 128
N_RECAL = 100_000
N_RETAIN_EVAL = 5_000
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
EVAL_TF = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])
TRAIN_TF = T.Compose([T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor(),
                      T.Normalize(MEAN, STD)])


class HFImages(Dataset):
    """A Hugging Face split, with labels remapped into the 1000-way index space."""

    def __init__(self, split, indices, transform, label_map):
        self.split, self.indices = split, [int(i) for i in indices]
        self.transform, self.label_map = transform, label_map

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        rec = self.split[self.indices[i]]
        return self.transform(rec["image"].convert("RGB")), self.label_map[int(rec["label"])]


def loader(ds, batch_size, shuffle, seed, workers):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=True, worker_init_fn=seed_worker,
                      generator=loader_generator(seed), persistent_workers=bool(workers))


def build_datasets(args):
    from datasets import load_dataset

    with open(CLASS_MAP) as f:
        cmap = json.load(f)
    lut = {int(k): int(v) for k, v in cmap["in100_to_1k"].items()}
    forget = int(cmap["forget_in100"])
    hf = load_dataset("clane9/imagenet-100")
    train, val = hf["train"], hf["validation"]
    tr, va = np.asarray(train["label"]), np.asarray(val["label"])
    f_tr, r_tr = np.where(tr == forget)[0], np.where(tr != forget)[0]
    f_va, r_va = np.where(va == forget)[0], np.where(va != forget)[0]
    rng = np.random.default_rng(args.seed)
    r_va = np.sort(rng.choice(r_va, size=min(args.n_retain_eval, len(r_va)), replace=False))
    r_recal = np.sort(rng.choice(r_tr, size=min(args.n_recal, len(r_tr)), replace=False))
    return {"forget_aug": HFImages(train, f_tr, TRAIN_TF, lut),
            "retain_aug": HFImages(train, r_tr, TRAIN_TF, lut),
            "forget_train": HFImages(train, f_tr, EVAL_TF, lut),
            "forget_val": HFImages(val, f_va, EVAL_TF, lut),
            "retain_val": HFImages(val, r_va, EVAL_TF, lut),
            "retain_recal": HFImages(train, r_recal, EVAL_TF, lut)}


def mixed_batches(data, args):
    """Yields (x, is_forget) with exactly the requested forget/retain split per batch."""
    n_r = int(round(args.retain_fraction * UNLEARN_BS))
    n_f = UNLEARN_BS - n_r

    def cycle(ds, n, seed):
        ld = loader(ds, n, True, seed, args.num_workers)
        while True:
            yield from ld

    forget, retain = cycle(data["forget_aug"], n_f, args.seed + 2), None
    if n_r:
        retain = cycle(data["retain_aug"], n_r, args.seed + 3)
    while True:
        xf = next(forget)[0]
        xs = [xf] + ([next(retain)[0]] if retain else [])
        n_r_got = xs[1].size(0) if retain else 0
        yield torch.cat(xs), torch.cat([torch.ones(xf.size(0), dtype=torch.bool),
                                        torch.zeros(n_r_got, dtype=torch.bool)])


def run_badteacher(model, data, args, device):
    """Forget samples follow a random network, retain samples a frozen copy. BN stays frozen."""
    competent = copy.deepcopy(model).to(device).eval().requires_grad_(False)
    torch.manual_seed(args.seed + 12345)
    incompetent = resnet50(weights=None).to(device).eval().requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")
    stream = mixed_batches(data, args)
    with bn_mode_ctx(model, "eval", log=log):
        for step in range(1, args.steps + 1):
            x, is_forget = next(stream)
            x, is_forget = x.to(device, non_blocking=True), is_forget.to(device)
            with torch.no_grad(), torch.amp.autocast("cuda"):
                teacher = torch.where(is_forget.view(-1, 1), incompetent(x).float(),
                                      competent(x).float())
            with torch.amp.autocast("cuda"):
                logits = model(x)
            loss = F.kl_div(F.log_softmax(logits.float(), 1), F.softmax(teacher, 1),
                            reduction="batchmean")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            assert_bn_mode(model, "eval")
            if step % 10 == 0 or step == 1:
                log(f"  [BadTeacher] step {step}/{args.steps} KL={loss.item():.4f}")
    del competent, incompetent
    torch.cuda.empty_cache()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--retain-fraction", type=float, default=0.9)
    ap.add_argument("--n-recal", type=int, default=N_RECAL)
    ap.add_argument("--n-retain-eval", type=int, default=N_RETAIN_EVAL)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out", default="imagenet100.json")
    args = ap.parse_args()

    set_seed(args.seed, deterministic=True)
    device = get_device(args.gpu)
    data = build_datasets(args)
    evals = {k: loader(data[k], 128, False, args.seed, args.num_workers)
             for k in ("forget_train", "forget_val", "retain_val")}

    def panel(model):
        return {k: accuracy(model, ld, device) for k, ld in evals.items()}

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device).eval()
    out = {"baseline": panel(model), "args": vars(args)}
    model = run_badteacher(model, data, args, device)
    out["pre"] = panel(model)
    unlearned = copy.deepcopy(model)

    # Shuffled: the dataset is stored in class order, and train-mode BN normalises each batch.
    recal = loader(data["retain_recal"], RECAL_BS, True, args.seed + 7, args.num_workers)
    model, info = recalibrate_bn(model, recal, device)
    assert info["weights_bit_identical"]
    out["post"] = panel(model)

    picks = np.random.default_rng(args.seed + 9091).choice(len(data["retain_recal"]), size=10,
                                                           replace=False)
    ten = DataLoader(Subset(data["retain_recal"], [int(i) for i in picks]), batch_size=10,
                     shuffle=False)
    unlearned, info = recalibrate_bn(unlearned, ten, device, strict=False)
    assert info["weights_bit_identical"]
    out["ten_images"] = panel(unlearned)

    log(f"forget(val): {out['pre']['forget_val']:.1f} -> {out['post']['forget_val']:.1f} "
        f"(ten images: {out['ten_images']['forget_val']:.1f}) | "
        f"retain(val): {out['pre']['retain_val']:.1f} -> {out['post']['retain_val']:.1f}")
    dump(out, args.out)


if __name__ == "__main__":
    main()
