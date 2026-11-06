"""CIFAR-10/100 loaders and forget/retain splits.

Split modes:
  'class'       - single-class forget. Forget = every training sample of `forget_class`.
  'classes'     - multi-class forget, given by `forget_classes`.
  'iid_random'  - forget = an i.i.d. random subset of the whole training set at fraction p.

`make_mixed_unlearn_loader` builds an unlearning loader that mixes forget and retain
samples at a chosen retain fraction.

Forget/retain accuracy is computed on the train split with the deterministic transform;
the test-split figures are reported separately.
"""

import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

from ..utils.seeding import loader_generator, seed_worker

DEFAULT_DATA_ROOT = os.environ.get("DATA_ROOT", "./data")

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def _stats(dataset_name):
    return (CIFAR100_MEAN, CIFAR100_STD) if dataset_name == "cifar100" else (
        CIFAR10_MEAN, CIFAR10_STD)


def train_transform(dataset_name="cifar10"):
    mean, std = _stats(dataset_name)
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def eval_transform(dataset_name="cifar10"):
    """Deterministic transform, used for every metric."""
    mean, std = _stats(dataset_name)
    return T.Compose([T.ToTensor(), T.Normalize(mean, std)])


def get_dataset(split, dataset_name="cifar10", augment=False,
                root=DEFAULT_DATA_ROOT, download=True):
    cls = torchvision.datasets.CIFAR100 if dataset_name == "cifar100" \
        else torchvision.datasets.CIFAR10
    tf = train_transform(dataset_name) if augment else eval_transform(dataset_name)
    return cls(root=root, train=(split == "train"), transform=tf, download=download)


def targets_of(dataset):
    return np.asarray(dataset.targets)


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #
def make_split(dataset, mode="class", forget_class=0, p=0.1, seed=0,
               forget_classes=None):
    """Return (forget_idx, retain_idx) as numpy int arrays over `dataset`.

    'class':      forget = all samples with label == forget_class
    'classes':    forget = all samples whose label is in `forget_classes`
    'iid_random': forget = an i.i.d. sample of round(p * N) indices across ALL classes
    """
    t = targets_of(dataset)
    n = len(t)
    if mode == "class":
        forget = np.where(t == forget_class)[0]
        retain = np.where(t != forget_class)[0]
    elif mode == "classes":
        fc = np.asarray(sorted(forget_classes))
        forget = np.where(np.isin(t, fc))[0]
        retain = np.where(~np.isin(t, fc))[0]
    elif mode == "iid_random":
        rng = np.random.default_rng(seed)
        k = int(round(p * n))
        forget = np.sort(rng.choice(n, size=k, replace=False))
        mask = np.ones(n, bool)
        mask[forget] = False
        retain = np.where(mask)[0]
    else:
        raise ValueError(f"unknown split mode {mode!r}")
    assert len(np.intersect1d(forget, retain)) == 0, "forget/retain overlap"
    assert len(forget) + len(retain) == n, "split does not partition the dataset"
    return forget, retain


def _loader(dataset, idx, batch_size, shuffle, seed, num_workers=4, drop_last=False):
    ds = Subset(dataset, [int(i) for i in idx]) if idx is not None else dataset
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=True, drop_last=drop_last,
        worker_init_fn=seed_worker, generator=loader_generator(seed),
        persistent_workers=bool(num_workers),
    )


def build_loaders(dataset_name="cifar10", mode="class", forget_class=0, p=0.1,
                  seed=0, batch_size=128, eval_batch_size=256, num_workers=4,
                  root=DEFAULT_DATA_ROOT, forget_classes=None,
                  unlearn_batch_size=256, recal_batch_size=128, datasets=None,
                  eval_num_workers=2):
    """Everything downstream needs, in one dict.

    Keys:
      train_aug            - full train set, augmented (base training)
      retain_train_aug     - retain subset, augmented (retrain baseline, fine-tuning)
      forget_train_aug     - forget subset, augmented (unlearning steps)
      retain_recal_aug     - retain subset, augmented (default BN recalibration stream)
      retain_recal         - retain subset, deterministic, fixed order
      forget_eval_train / retain_eval_train  - train split, deterministic
      forget_eval_test  / retain_eval_test   - test split, deterministic
      test_full            - whole test set
      idx                  - the raw index arrays

    `datasets` supplies (train_aug, train_det, test_det) for a dataset other than CIFAR.
    """
    if datasets is not None:
        tr_aug, tr_det, te_det = datasets
    else:
        tr_aug = get_dataset("train", dataset_name, augment=True, root=root)
        tr_det = get_dataset("train", dataset_name, augment=False, root=root)
        te_det = get_dataset("test", dataset_name, augment=False, root=root)

    f_idx, r_idx = make_split(tr_det, mode=mode, forget_class=forget_class, p=p,
                              seed=seed, forget_classes=forget_classes)

    if mode in ("class", "classes"):
        t_te = targets_of(te_det)
        if mode == "class":
            f_te = np.where(t_te == forget_class)[0]
            r_te = np.where(t_te != forget_class)[0]
        else:
            fc = np.asarray(sorted(forget_classes))
            f_te = np.where(np.isin(t_te, fc))[0]
            r_te = np.where(~np.isin(t_te, fc))[0]
    else:
        # i.i.d. forget set has no test-set counterpart by class; use the whole test
        # set for retain-side utility and leave the forget-side test split empty.
        f_te = np.array([], dtype=int)
        r_te = np.arange(len(te_det))

    return {
        "train_aug": _loader(tr_aug, None, batch_size, True, seed, num_workers),
        # Unlearning runs at `unlearn_batch_size`, recalibration at `recal_batch_size`.
        "retain_train_aug": _loader(tr_aug, r_idx, unlearn_batch_size, True, seed + 1,
                                    num_workers, drop_last=True),
        "forget_train_aug": _loader(tr_aug, f_idx, unlearn_batch_size, True, seed + 2,
                                    num_workers, drop_last=True),
        # Default recalibration stream. Training consumes the augmented retain
        # distribution, so phi* is estimated on the same one.
        "retain_recal_aug": _loader(tr_aug, r_idx, recal_batch_size, True, seed + 5,
                                    num_workers, drop_last=True),
        # Deterministic and unshuffled: `update_bn` is exactly repeatable only at a fixed
        # batch partition.
        "retain_recal": _loader(tr_det, r_idx, recal_batch_size, False, seed + 3,
                                num_workers),
        "forget_eval_train": _loader(tr_det, f_idx, eval_batch_size, False, seed,
                                     eval_num_workers),
        "retain_eval_train": _loader(tr_det, r_idx, eval_batch_size, False, seed,
                                     eval_num_workers),
        "forget_eval_test": _loader(te_det, f_te, eval_batch_size, False, seed,
                                    eval_num_workers)
        if len(f_te) else None,
        "retain_eval_test": _loader(te_det, r_te, eval_batch_size, False, seed,
                                    eval_num_workers),
        "test_full": _loader(te_det, None, eval_batch_size, False, seed, eval_num_workers),
        "idx": {"forget_train": f_idx, "retain_train": r_idx,
                "forget_test": f_te, "retain_test": r_te},
        "datasets": {"train_aug": tr_aug, "train_det": tr_det, "test_det": te_det},
        "config": {"dataset": dataset_name, "mode": mode, "forget_class": forget_class,
                   "forget_classes": (sorted(forget_classes) if forget_classes
                                      else None),
                   "p": p, "seed": seed, "batch_size": batch_size,
                   "unlearn_batch_size": unlearn_batch_size,
                   "recal_batch_size": recal_batch_size,
                   "n_forget_train": int(len(f_idx)), "n_retain_train": int(len(r_idx))},
    }


# --------------------------------------------------------------------------- #
# mixed forget/retain unlearning loader
# --------------------------------------------------------------------------- #
class MixedBatchLoader:
    """Yields batches with a fixed fraction of retain samples.

    retain_fraction = 0.0  -> forget-only
    retain_fraction = 0.9  -> the natural class ratio for single-class CIFAR-10 forgetting
    Each batch has exactly `round(retain_fraction * batch_size)` retain samples and the
    remainder forget samples. Iterates indefinitely; callers take `steps` batches.
    """

    def __init__(self, forget_loader, retain_loader, retain_fraction, batch_size):
        self.fl, self.rl = forget_loader, retain_loader
        self.frac = float(retain_fraction)
        self.bs = batch_size
        self.n_r = int(round(self.frac * batch_size))
        self.n_f = batch_size - self.n_r
        self._fi, self._ri = iter(forget_loader), iter(retain_loader)

    def _next(self, which):
        it, ld = (self._fi, self.fl) if which == "f" else (self._ri, self.rl)
        try:
            b = next(it)
        except StopIteration:
            it = iter(ld)
            b = next(it)
        if which == "f":
            self._fi = it
        else:
            self._ri = it
        return b

    def _take(self, which, n):
        """Accumulate exactly n samples from the given loader."""
        xs, ys = [], []
        got = 0
        while got < n:
            x, y = self._next(which)
            take = min(n - got, x.size(0))
            xs.append(x[:take])
            ys.append(y[:take])
            got += take
        return torch.cat(xs), torch.cat(ys)

    def __iter__(self):
        return self

    def __next__(self):
        parts_x, parts_y, is_forget = [], [], []
        if self.n_f > 0:
            xf, yf = self._take("f", self.n_f)
            parts_x.append(xf)
            parts_y.append(yf)
            is_forget.append(torch.ones(self.n_f, dtype=torch.bool))
        if self.n_r > 0:
            xr, yr = self._take("r", self.n_r)
            parts_x.append(xr)
            parts_y.append(yr)
            is_forget.append(torch.zeros(self.n_r, dtype=torch.bool))
        return torch.cat(parts_x), torch.cat(parts_y), torch.cat(is_forget)


def make_mixed_unlearn_loader(loaders, retain_fraction, batch_size=128):
    return MixedBatchLoader(loaders["forget_train_aug"], loaders["retain_train_aug"],
                            retain_fraction, batch_size)
