"""LP-50 and NCC: feature-space probes of what the weights still encode.

LP-50: penultimate (512-d) features -> LogisticRegression(solver='lbfgs', max_iter=1000),
50 train-split samples per class, forget-class accuracy on the test split.

NCC: class-mean features on 50 samples per class, l2 nearest-centroid assignment.

Feature extraction always runs with `model.eval()`.
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from ..models.resnet_cifar import extract_features


def _balanced_subset(labels, n_per_class, num_classes, rng):
    idx = []
    for c in range(num_classes):
        pool = np.where(labels == c)[0]
        if len(pool) == 0:
            continue
        take = min(n_per_class, len(pool))
        idx.append(rng.choice(pool, size=take, replace=False))
    return np.concatenate(idx) if idx else np.array([], dtype=int)


def _forget_mask(labels, forget_class):
    """Labels in the forget class(es); empty when `forget_class` is None (instance-wise)."""
    return np.isin(labels, [] if forget_class is None else forget_class)


def linear_probe_50(model, loaders, device, forget_class=0, n_per_class=50,
                    num_classes=10, seed=0, max_iter=1000):
    """Fit LR on n_per_class train-split samples/class; report accuracy on the test split.

    Returns forget-class accuracy (the headline LP-50 number), retain-class accuracy,
    and overall accuracy. `forget_class` is a class, a list of classes, or None.
    """
    rng = np.random.default_rng(seed)

    # Fit set: train split, deterministic transform, balanced over ALL classes.
    f_tr, y_f_tr = extract_features(model, loaders["forget_eval_train"], device)
    r_tr, y_r_tr = extract_features(model, loaders["retain_eval_train"], device)
    X_tr = torch.cat([f_tr, r_tr]).numpy()
    y_tr = torch.cat([y_f_tr, y_r_tr]).numpy()

    sel = _balanced_subset(y_tr, n_per_class, num_classes, rng)
    X_fit, y_fit = X_tr[sel], y_tr[sel]

    # Eval set: full test split.
    X_te, y_te = extract_features(model, loaders["test_full"], device)
    X_te, y_te = X_te.numpy(), y_te.numpy()

    clf = LogisticRegression(solver="lbfgs", max_iter=max_iter)
    clf.fit(X_fit, y_fit)
    pred = clf.predict(X_te)

    fmask = _forget_mask(y_te, forget_class)
    rmask = ~fmask
    return {
        "lp50_forget_acc": 100.0 * float((pred[fmask] == y_te[fmask]).mean())
        if fmask.any() else float("nan"),
        "lp50_retain_acc": 100.0 * float((pred[rmask] == y_te[rmask]).mean())
        if rmask.any() else float("nan"),
        "lp50_overall_acc": 100.0 * float((pred == y_te).mean()),
        "n_fit": int(len(y_fit)),
        "n_per_class": n_per_class,
        "classes_fit": sorted(int(c) for c in np.unique(y_fit)),
    }


def ncc(model, loaders, device, forget_class=0, n_per_class=50, num_classes=10, seed=0):
    """Nearest-class-centroid in feature space, l2 distance.

    Centroids from n_per_class train-split samples per class; assignment on the test split.
    """
    rng = np.random.default_rng(seed)

    f_tr, y_f_tr = extract_features(model, loaders["forget_eval_train"], device)
    r_tr, y_r_tr = extract_features(model, loaders["retain_eval_train"], device)
    X_tr = torch.cat([f_tr, r_tr])
    y_tr = torch.cat([y_f_tr, y_r_tr])

    sel = _balanced_subset(y_tr.numpy(), n_per_class, num_classes, rng)
    X_fit, y_fit = X_tr[sel], y_tr[sel]

    classes = sorted(int(c) for c in torch.unique(y_fit))
    centroids = torch.stack([X_fit[y_fit == c].mean(0) for c in classes])

    X_te, y_te = extract_features(model, loaders["test_full"], device)
    d = torch.cdist(X_te, centroids)                 # [N, C] l2 distances
    pred = torch.tensor(classes)[d.argmin(1)]

    fmask = torch.from_numpy(_forget_mask(y_te.numpy(), forget_class))
    rmask = ~fmask
    return {
        "ncc_forget_acc": 100.0 * float((pred[fmask] == y_te[fmask]).float().mean())
        if fmask.any() else float("nan"),
        "ncc_retain_acc": 100.0 * float((pred[rmask] == y_te[rmask]).float().mean())
        if rmask.any() else float("nan"),
        "ncc_overall_acc": 100.0 * float((pred == y_te).float().mean()),
        "n_per_class": n_per_class,
        "classes": classes,
    }
