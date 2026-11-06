"""Membership-inference attacks: loss threshold (Yeom et al.), confidence and modified
entropy (Song & Mittal), and final-layer gradient norm (Nasr et al.).

Members = forget-class train samples, non-members = forget-class test samples. Scores are
oriented so that higher means "more likely a member". Reports AUC, TPR@FPR=0.01 and
TPR@FPR=0.001 per attack.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

from ..models.resnet_cifar import FeatureExtractor

ATTACKS = ["loss_threshold", "confidence", "modified_entropy", "grad_norm"]


@torch.no_grad()
def _collect(model, loader, device):
    """Per-sample logits, features and labels, with the model in eval mode."""
    if loader is None:
        return None
    was_training = model.training
    model.eval()
    fx = FeatureExtractor(model)
    logits, feats, labels = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        f = fx(x)
        feats.append(f.cpu())
        logits.append(model.fc(f).cpu())
        labels.append(y.cpu())
    model.train(was_training)
    return (torch.cat(logits).float(), torch.cat(feats).float(), torch.cat(labels))


def _modified_entropy(probs, labels, eps=1e-12):
    """Song & Mittal's Mentr, negated so that higher = more member-like.

    Mentr(x,y) = -(1-p_y) log(p_y) - sum_{i != y} p_i log(1 - p_i)
    Low Mentr means confident-and-correct, i.e. member-like, so we return -Mentr.
    """
    p = probs.clamp(eps, 1 - eps)
    n, c = p.shape
    onehot = F.one_hot(labels, c).bool()
    py = p[onehot]
    term_y = -(1 - py) * torch.log(py)
    log1m = torch.log(1 - p)
    term_all = -(p * log1m)
    term_other = term_all.sum(1) - term_all[onehot]
    return -(term_y + term_other)


def _scores(model, loader, device):
    """All four membership scores for one loader. Higher = more member-like."""
    got = _collect(model, loader, device)
    if got is None:
        return None
    logits, feats, labels = got
    probs = F.softmax(logits, 1)
    loss = F.cross_entropy(logits, labels, reduction="none")
    conf = probs.max(1).values

    # Nasr et al. final-layer gradient norm, computed in closed form.
    onehot = F.one_hot(labels, probs.shape[1]).float()
    dlogits = (probs - onehot).norm(dim=1)          # ||softmax - onehot||_2
    fnorm = feats.norm(dim=1)                       # ||penultimate feature||_2
    gnorm = dlogits * fnorm                         # ||grad W|| (Frobenius, rank-1)

    return {
        "loss_threshold": (-loss).numpy(),          # low loss -> member
        "confidence": conf.numpy(),                 # high confidence -> member
        "modified_entropy": _modified_entropy(probs, labels).numpy(),
        "grad_norm": (-gnorm).numpy(),              # small gradient -> member
        "_n": int(labels.numel()),
    }


def _tpr_at_fpr(y_true, y_score, target_fpr):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(idx, 0)
    return float(tpr[idx])


def run_mia(model, loaders, device, member_loader=None, nonmember_loader=None):
    """Run all four attacks. Returns {attack: {auc, tpr@1%, tpr@0.1%, n_*}}.

    Defaults: members = forget-class train split, non-members = forget-class test split.
    """
    member_loader = member_loader or loaders["forget_eval_train"]
    nonmember_loader = nonmember_loader or loaders["forget_eval_test"]
    if member_loader is None or nonmember_loader is None:
        return {"error": "member or non-member loader unavailable for this split mode"}

    s_in = _scores(model, member_loader, device)
    s_out = _scores(model, nonmember_loader, device)

    out = {}
    for a in ATTACKS:
        y = np.concatenate([np.ones(s_in["_n"]), np.zeros(s_out["_n"])])
        s = np.concatenate([s_in[a], s_out[a]])
        finite = np.isfinite(s)
        y, s = y[finite], s[finite]
        out[a] = {
            "auc": float(roc_auc_score(y, s)),
            "tpr_at_fpr_0.01": _tpr_at_fpr(y, s, 0.01),
            "tpr_at_fpr_0.001": _tpr_at_fpr(y, s, 0.001),
            "n_members": int(y.sum()),
            "n_nonmembers": int((1 - y).sum()),
        }
    out["_setup"] = {
        "members": "forget-class TRAIN samples",
        "nonmembers": "forget-class TEST samples",
        "n_members": s_in["_n"],
        "n_nonmembers": s_out["_n"],
    }
    return out
