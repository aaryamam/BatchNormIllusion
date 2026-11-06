"""Relearn-AUC: how fast the forget class comes back under a small fine-tune.

Fine-tune on 50 forget-class samples for 50 epochs, record the forget-accuracy curve, and
integrate accuracy/100 over normalised epoch with the trapezoid rule, so ReAUC is in
[0, 1]. The fine-tune works on a deep copy and runs in ordinary train mode.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from ..utils.seeding import loader_generator, seed_worker
from .accuracy import accuracy


def _forget_finetune_loader(loaders, n_samples=50, batch_size=50, seed=0,
                            num_workers=0, per_class=None):
    """`n_samples` forget-class TRAIN samples, augmented (the attacker's data).

    `per_class` instead draws exactly that many samples from EACH forget class. The
    multi-class condition (CIFAR-100, five classes) uses 50 per forget class
    rather than 50 in total, and uniform sampling over the pooled forget set would only
    hit that in expectation.
    """
    ds_aug = loaders["datasets"]["train_aug"]
    f_idx = loaders["idx"]["forget_train"]
    rng = np.random.default_rng(seed)
    if per_class:
        targets = np.asarray(loaders["datasets"]["train_det"].targets)[f_idx]
        chosen = []
        for c in np.unique(targets):
            pool = f_idx[targets == c]
            take = rng.choice(len(pool), size=min(per_class, len(pool)), replace=False)
            chosen.extend(int(pool[i]) for i in take)
    else:
        take = rng.choice(len(f_idx), size=min(n_samples, len(f_idx)), replace=False)
        chosen = [int(f_idx[i]) for i in take]
    return DataLoader(
        Subset(ds_aug, chosen), batch_size=min(batch_size, len(chosen)),
        shuffle=True, num_workers=num_workers, drop_last=False,
        worker_init_fn=seed_worker, generator=loader_generator(seed),
    ), chosen


def relearn_auc(model, loaders, device, n_samples=50, epochs=50, lr=1e-4,
                batch_size=50, seed=0, log=None, per_class=None):
    """Returns {'relearn_auc': float, 'curve': [...], 'epochs': ...}.

    The curve's epoch 0 entry is the pre-fine-tune forget accuracy, so the caller can
    see the starting point; the AUC integrates epochs 0..N over normalised x in [0,1].
    """
    student = copy.deepcopy(model).to(device)
    loader, chosen = _forget_finetune_loader(loaders, n_samples, batch_size,
                                            seed, per_class=per_class)
    opt = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9)
    crit = nn.CrossEntropyLoss()

    eval_loader = loaders["forget_eval_test"] or loaders["forget_eval_train"]
    curve = [accuracy(student, eval_loader, device)]

    for ep in range(epochs):
        student.train()
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            crit(student(x), y).backward()
            opt.step()
        curve.append(accuracy(student, eval_loader, device))
        if log and (ep + 1) % 25 == 0:
            log(f"  [relearn] epoch {ep+1}/{epochs} forget_acc={curve[-1]:.2f}")

    y = np.asarray(curve, dtype=float) / 100.0
    x = np.linspace(0.0, 1.0, len(y))
    auc = float(np.trapezoid(y, x)) if hasattr(np, "trapezoid") else float(np.trapz(y, x))

    del student
    torch.cuda.empty_cache()
    return {
        "relearn_auc": auc,
        "curve": [float(c) for c in curve],
        "epochs": epochs,
        "n_samples": int(len(chosen)),
        "lr": lr,
        "eval_split": "test" if loaders["forget_eval_test"] is not None else "train",
    }
