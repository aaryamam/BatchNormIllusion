"""Shared helpers for the experiment scripts: paths, model loading, one measurement pass.

For each checkpoint: measure Pre-F / Pre-R, recalibrate BN on retain data, measure
Post-F / Post-R. Both the pre- and post-recalibration checkpoints are saved.
"""

import ast
import copy
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.cifar import build_loaders  # noqa: E402
from src.data.tiny_imagenet import tiny_datasets  # noqa: E402
from src.diagnostic.recalibrate import bn_buffer_maxdiff, recalibrate_bn  # noqa: E402
from src.eval.accuracy import evaluate_all  # noqa: E402
from src.eval.linear_probe import linear_probe_50, ncc  # noqa: E402
from src.eval.relearn import relearn_auc  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.models.resnet_cifar import bn_state  # noqa: E402
from src.unlearn.registry import EVAL_MODE_METHODS, run_method  # noqa: E402

CKPT_DIR = "checkpoints"
RESULT_DIR = "results"
NUM_CLASSES = {"cifar10": 10, "cifar100": 100, "tiny_imagenet": 200}


def get_device(gpu=0):
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        return f"cuda:{gpu}"
    return "cpu"


def log(msg):
    print(msg, flush=True)


def load_state(path):
    return torch.load(path, map_location="cpu", weights_only=False)["model"]


def ckpt_name(which, dataset="cifar10", arch="resnet18", classes="class0", seed=0):
    """Checkpoint filename, e.g. tiny_imagenet_resnet50_base_class0_seed0.pt."""
    ds = "" if dataset == "cifar10" else f"{dataset}_"
    ar = "" if arch == "resnet18" else f"{arch}_"
    return f"{ds}{ar}{which}_{classes}_seed{seed}.pt"


def load_base(dataset="cifar10", arch="resnet18"):
    return load_state(os.path.join(CKPT_DIR, ckpt_name("base", dataset, arch)))


def load_retrain(dataset="cifar10", arch="resnet18", seed=0):
    return load_state(os.path.join(CKPT_DIR, ckpt_name("retrain", dataset, arch, seed=seed)))


def fresh_model(state, dataset="cifar10", device="cuda:0", arch="resnet18"):
    model = build_model(arch, NUM_CLASSES[dataset]).to(device)
    model.load_state_dict(state)
    return model


def make_loaders(dataset="cifar10", forget_class=0, seed=0, mode="class", p=0.1,
                 batch_size=128, forget_classes=None, arch="resnet18", num_workers=4,
                 unlearn_batch_size=256):
    extra = {}
    if dataset == "tiny_imagenet":
        extra = {"datasets": tiny_datasets(arch), "eval_num_workers": num_workers}
    return build_loaders(dataset_name=dataset, mode=mode, forget_class=forget_class,
                         forget_classes=forget_classes, p=p, seed=seed,
                         batch_size=batch_size, num_workers=num_workers,
                         unlearn_batch_size=unlearn_batch_size, **extra)


def measure(model, loaders, device, forget_class=0, num_classes=10, seed=0,
            do_probe=False, do_relearn=False, relearn_lr=1e-4):
    """The metric panel for one model state."""
    out = dict(evaluate_all(model, loaders, device))
    if do_probe:
        out.update(linear_probe_50(model, loaders, device, forget_class=forget_class,
                                   num_classes=num_classes, seed=seed))
        out.update(ncc(model, loaders, device, forget_class=forget_class,
                       num_classes=num_classes, seed=seed))
    if do_relearn:
        out.update(relearn_auc(model, loaders, device, seed=seed, lr=relearn_lr))
    return out


def summarize(method, seed, tag, pre, post, **extra):
    return {"method": method, "seed": seed, "tag": tag,
            "PreF": pre["forget_train"], "PostF": post["forget_train"],
            "dF": post["forget_train"] - pre["forget_train"],
            "PreR": pre["retain_train"], "PostR": post["retain_train"],
            "pre": pre, "post": post, **extra}


def save_pair(pre_state, post_state, subdir, tag):
    path = os.path.join(CKPT_DIR, subdir)
    os.makedirs(path, exist_ok=True)
    torch.save({"model": pre_state}, os.path.join(path, f"{tag}_pre.pt"))
    torch.save({"model": post_state}, os.path.join(path, f"{tag}_post.pt"))


def run_checkpoint(method, base_state, loaders, device, seed=0, forget_class=0,
                   dataset="cifar10", overrides=None, tag=None, subdir="table1",
                   do_probe=False, do_relearn=False, arch="resnet18",
                   recal_stream="retain_recal_aug", bn_mode=None, relearn_lr=1e-4):
    """Unlearn -> measure -> recalibrate -> measure. Returns one result row.

    `method="retrain"` skips unlearning and measures `base_state` as it stands. `bn_mode`
    runs a method outside its declared BN mode.
    """
    t0 = time.time()
    tag = tag or f"{method}_seed{seed}"
    nc = NUM_CLASSES[dataset]
    model = fresh_model(base_state, dataset, device, arch)
    phi0 = bn_state(model)

    info = {"bn_mode_used": "n/a"}
    if method != "retrain":
        model, info = run_method(method, model, loaders, device, overrides=overrides,
                                 bn_mode=bn_mode, allow_bn_override=bn_mode is not None,
                                 seed=seed, log=log, inplace=True)

    bn_maxdiff = bn_buffer_maxdiff(phi0, bn_state(model))["max_abs_diff_overall"]
    if method in EVAL_MODE_METHODS or bn_mode == "eval":
        assert bn_maxdiff == 0.0, f"{method}: BN moved in eval mode"

    pre = measure(model, loaders, device, forget_class, nc, seed, do_probe, do_relearn,
                  relearn_lr)
    pre_state = copy.deepcopy(model.state_dict())

    model, recal = recalibrate_bn(model, loaders[recal_stream], device, inplace=True)
    post = measure(model, loaders, device, forget_class, nc, seed, do_probe, do_relearn,
                   relearn_lr)

    save_pair(pre_state, model.state_dict(), subdir, tag)
    row = summarize(method, seed, tag, pre, post, bn_mode=info["bn_mode_used"],
                    bn_maxdiff=bn_maxdiff, hparams=info.get("hparams", {}),
                    weights_bit_identical=recal["weights_bit_identical"],
                    bn_drift=[d["delta_mu_l2_over_sigma"]
                              for d in recal.get("drift", {}).values()],
                    secs=time.time() - t0)
    log(f"  {tag}: PreF={row['PreF']:.2f} PostF={row['PostF']:.2f} dF={row['dF']:+.2f} "
        f"PreR={row['PreR']:.2f} PostR={row['PostR']:.2f}")
    del model
    torch.cuda.empty_cache()
    return row


def parse_overrides(items):
    """['ga.lr=1e-4', 'ga.steps=50'] -> {'ga': {'lr': 0.0001, 'steps': 50}}."""
    out = {}
    for item in items:
        key, value = item.split("=", 1)
        method, name = key.split(".", 1)
        out.setdefault(method, {})[name] = ast.literal_eval(value)
    return out


def read_results(name):
    with open(os.path.join(RESULT_DIR, name)) as f:
        return json.load(f)


def dump(obj, name):
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    return path
