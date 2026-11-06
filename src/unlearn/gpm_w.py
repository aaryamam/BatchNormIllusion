"""GPM-W: weight-projection gradient ascent. BN mode: train.

For each Conv2d except the stem, the uncentred covariance of the layer's input patches over
`samples_per_class` retain samples per class gives a basis U_k holding `svd_threshold` of
the spectrum. Each step computes the forget CE gradient, projects the conv gradients off that
subspace (G <- G - (G U_k) U_k^T), and ascends on all parameters.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .common import StepBudget, assert_bn_mode, bn_mode_ctx

BN_MODE = "train"
DEFAULTS = {"lr": 1e-3, "steps": 200, "svd_threshold": 0.95, "samples_per_class": 500,
            "batch_size": 128}


def _retain_bases(model, loaders, device, threshold, samples_per_class, batch_size, log):
    ds = loaders["datasets"]["train_det"]
    targets = np.asarray(ds.targets)
    r_idx = loaders["idx"]["retain_train"]
    rng = np.random.default_rng(0)
    chosen = []
    for c in np.unique(targets[r_idx]):
        pool = r_idx[targets[r_idx] == c]
        chosen.extend(rng.choice(pool, size=min(samples_per_class, len(pool)),
                                 replace=False).tolist())
    loader = DataLoader(Subset(ds, [int(i) for i in chosen]), batch_size=batch_size,
                        shuffle=False, num_workers=4)

    convs = [(n, m) for n, m in model.named_modules()
             if isinstance(m, nn.Conv2d) and n != "conv1"]
    inputs = {}

    def capture(name):
        def hook(_mod, inp, _out):
            inputs[name] = inp[0].detach()
        return hook

    hooks = [m.register_forward_hook(capture(n)) for n, m in convs]
    model.eval()
    covar = {n: None for n, _ in convs}
    with torch.no_grad():
        for x, _y in loader:
            model(x.to(device, non_blocking=True))
            for n, m in convs:
                patch = F.unfold(inputs[n], m.kernel_size, dilation=m.dilation,
                                 padding=m.padding, stride=m.stride)
                patch = patch.permute(0, 2, 1).reshape(-1, patch.shape[1]).double()
                c = patch.t() @ patch
                covar[n] = c if covar[n] is None else covar[n] + c
    for h in hooks:
        h.remove()

    bases = {}
    for n, _ in convs:
        U, S, _V = torch.linalg.svd(covar[n])
        k = int(torch.sum((torch.cumsum(S, 0) / torch.sum(S)) <= threshold).item()) + 1
        bases[n] = U[:, :k].float()
        if log:
            log(f"  [GPM-W] {n}: k={k}/{S.shape[0]} ({100 * k / S.shape[0]:.1f}%)")
    return bases, convs


def unlearn_gpm_w(model, loaders, device, lr=1e-3, steps=200, svd_threshold=0.95,
                  samples_per_class=500, batch_size=128, bn_mode=BN_MODE, seed=0,
                  log=None, **_):
    bases, convs = _retain_bases(model, loaders, device, svd_threshold, samples_per_class,
                                 batch_size, log)
    crit = nn.CrossEntropyLoss()
    history = []
    with bn_mode_ctx(model, bn_mode, log=log):
        for step, (x, y) in StepBudget(loaders["forget_train_aug"], steps):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            model.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            with torch.no_grad():
                for n, mod in convs:
                    U = bases[n]
                    W = mod.weight.grad
                    G = W.reshape(W.shape[0], -1)
                    if G.shape[1] == U.shape[0]:
                        mod.weight.grad = (G - (G @ U) @ U.t()).reshape(W.shape)
                for p in model.parameters():
                    if p.grad is not None:
                        p.data += lr * p.grad
            assert_bn_mode(model, bn_mode)
            history.append({"step": step, "forget_ce": float(loss.item())})
            if log and (step % 50 == 0 or step == 1):
                log(f"  [GPM-W] step {step}/{steps} forget_CE={loss.item():.4f}")
    return model, {"history": history, "bn_mode": bn_mode,
                   "hparams": {"lr": lr, "steps": steps, "svd_threshold": svd_threshold,
                               "samples_per_class": samples_per_class}}
