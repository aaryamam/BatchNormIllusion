#!/usr/bin/env python3
"""Evaluate saved checkpoints: Relearn-AUC, LP-50, NCC and the LP decomposition.

    BN-residual = LP(M) - LP(M_cal)
    NC-residual = LP(M_cal) - LP(M_retrain)

Also computes the feature-space cross-correlation and the rank correlations
between forget accuracy and the leakage measures. Run scripts/run_unlearning.py first.

    python scripts/run_evaluation.py --gpu 0
"""

import argparse
import os
import statistics as st

import numpy as np
import torch
from scipy import stats

from common import (CKPT_DIR, dump, fresh_model, get_device, load_base, load_state, log,
                    make_loaders, read_results)
from src.diagnostic.recalibrate import recalibrate_bn
from src.eval.linear_probe import linear_probe_50, ncc
from src.eval.relearn import relearn_auc
from src.models.resnet_cifar import extract_features
from src.utils.seeding import set_seed


def load_model(tag, state, device):
    path = os.path.join(CKPT_DIR, "table1", f"{tag}_{state}.pt")
    return fresh_model(load_state(path), "cifar10", device)


def forget_features(model, loaders, device):
    feats, _ = extract_features(model, loaders["forget_eval_train"], device)
    return feats.double().flatten().cpu()


def correlation(x, y):
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    kt, kp = stats.kendalltau(x, y)
    return {"pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_rho": float(sr), "spearman_p": float(sp),
            "kendall_tau": float(kt), "kendall_p": float(kp)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--input", default="unlearning.json")
    ap.add_argument("--out", default="evaluation.json")
    args = ap.parse_args()

    device = get_device(args.gpu)
    rows = read_results(args.input)["rows"]
    base_state = load_base("cifar10")
    loaders_by_seed, base_h = {}, {}

    measured, features = [], []
    for r in rows:
        seed = r["seed"]
        if seed not in loaders_by_seed:
            set_seed(seed)
            loaders_by_seed[seed] = make_loaders("cifar10", forget_class=0, seed=seed)
        loaders = loaders_by_seed[seed]

        h = {}
        for state in ("pre", "post"):
            model = load_model(r["tag"], state, device)
            lp = linear_probe_50(model, loaders, device, forget_class=0, seed=seed)
            nc = ncc(model, loaders, device, forget_class=0, seed=seed)
            auc = relearn_auc(model, loaders, device, seed=seed)["relearn_auc"]
            h[state] = forget_features(model, loaders, device)
            measured.append({"method": r["method"], "seed": seed, "state": state,
                             "LP50": lp["lp50_forget_acc"], "NCC": nc["ncc_forget_acc"],
                             "ReAUC": auc})
            log(f"  {r['tag']:24s} {state:4s} LP50={lp['lp50_forget_acc']:6.2f} "
                f"NCC={nc['ncc_forget_acc']:6.2f} ReAUC={auc:.3f}")
            del model
            torch.cuda.empty_cache()

        if r["method"] == "retrain":
            continue
        if seed not in base_h:
            base = fresh_model(base_state, "cifar10", device)
            base, _ = recalibrate_bn(base, loaders["retain_recal_aug"], device, inplace=True)
            base_h[seed] = forget_features(base, loaders, device)
            del base
        d_w, d_bn = h["post"] - base_h[seed], h["pre"] - h["post"]
        e_w, e_bn = float(d_w.pow(2).sum()), float(d_bn.pow(2).sum())
        e_total = float((d_w + d_bn).pow(2).sum())
        features.append({"method": r["method"], "seed": seed, "bn_mode": r.get("bn_mode"),
                         "E_W": e_w, "E_BN": e_bn, "E_total": e_total,
                         "rho": float(torch.corrcoef(torch.stack([d_w, d_bn]))[0, 1]),
                         "pyth_error": abs(e_w + e_bn - e_total) / max(e_total, 1e-12)})

    by_key = {(m["method"], m["seed"], m["state"]): m for m in measured}
    decomposition = []
    for r in rows:
        ref = by_key.get(("retrain", r["seed"], "post"))
        if ref is None:
            continue
        pre, post = by_key[(r["method"], r["seed"], "pre")], by_key[(r["method"], r["seed"], "post")]
        decomposition.append({"method": r["method"], "seed": r["seed"],
                              "LP_pre": pre["LP50"], "LP_post": post["LP50"],
                              "LP_retrain": ref["LP50"],
                              "BN_residual": pre["LP50"] - post["LP50"],
                              "NC_residual": post["LP50"] - ref["LP50"]})

    methods = sorted({r["method"] for r in rows})

    def mean(items, key, method, **match):
        return st.mean(i[key] for i in items if i["method"] == method
                       and all(i[k] == v for k, v in match.items()))

    correlations = {}
    if len(methods) >= 3:
        post = {"state": "post"}
        for fk in ("PreF", "PostF"):
            x = np.array([mean(rows, fk, m) for m in methods])
            for yk in ("ReAUC", "LP50"):
                y = np.array([mean(measured, yk, m, **post) for m in methods])
                correlations[f"{fk}_vs_{yk}"] = correlation(x, y)

    dump({"measurements": measured, "decomposition": decomposition,
          "feature_analysis": features, "correlations": correlations}, args.out)


if __name__ == "__main__":
    main()
