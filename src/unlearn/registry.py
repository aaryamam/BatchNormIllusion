"""name -> (fn, hyperparameters, bn_mode). `run_method` enforces the declared BN mode."""

import copy

from .badteacher import BN_MODE as BT_BN, DEFAULTS as BT_D, unlearn_badteacher
from .ga import BN_MODE as GA_BN, DEFAULTS as GA_D, unlearn_ga
from .ga_ft import BN_MODE as GAFT_BN, DEFAULTS as GAFT_D, unlearn_ga_ft
from .gpm_w import BN_MODE as GPMW_BN, DEFAULTS as GPMW_D, unlearn_gpm_w
from .incomp_teacher import (BN_MODE as IT_BN, DEFAULTS as IT_D,
                             unlearn_incomp_teacher)
from .noise_inject import BN_MODE as NI_BN, DEFAULTS as NI_D, unlearn_noise_inject
from .random_labeling import BN_MODE as RL_BN, DEFAULTS as RL_D, unlearn_random_labeling
from .salun import BN_MODE as SL_BN, DEFAULTS as SL_D, unlearn_salun
from .scrub import BN_MODE as SC_BN, DEFAULTS as SC_D, unlearn_scrub
from .ssd import BN_MODE as SSD_BN, DEFAULTS as SSD_D, unlearn_ssd

REGISTRY = {
    "ga": {"fn": unlearn_ga, "hparams": dict(GA_D), "bn_mode": GA_BN},
    "ga_ft": {"fn": unlearn_ga_ft, "hparams": dict(GAFT_D), "bn_mode": GAFT_BN},
    "salun": {"fn": unlearn_salun, "hparams": dict(SL_D), "bn_mode": SL_BN},
    "scrub": {"fn": unlearn_scrub, "hparams": dict(SC_D), "bn_mode": SC_BN},
    "badteacher": {"fn": unlearn_badteacher, "hparams": dict(BT_D), "bn_mode": BT_BN},
    "incompteacher": {"fn": unlearn_incomp_teacher, "hparams": dict(IT_D),
                      "bn_mode": IT_BN},
    "noise_inject": {"fn": unlearn_noise_inject, "hparams": dict(NI_D),
                     "bn_mode": NI_BN},
    "ssd": {"fn": unlearn_ssd, "hparams": dict(SSD_D), "bn_mode": SSD_BN},
    "random_labeling": {"fn": unlearn_random_labeling, "hparams": dict(RL_D),
                        "bn_mode": RL_BN},
    "gpm_w": {"fn": unlearn_gpm_w, "hparams": dict(GPMW_D), "bn_mode": GPMW_BN},
}

# Methods that run with BN frozen.
EVAL_MODE_METHODS = {"badteacher"}


def available():
    return sorted(REGISTRY)


def get(name):
    if name not in REGISTRY:
        raise KeyError(f"unknown method {name!r}; available: {available()}")
    return REGISTRY[name]


def run_method(name, model, loaders, device, overrides=None, bn_mode=None,
               allow_bn_override=False, seed=0, log=None, inplace=False):
    """Run an unlearning method under its declared BN mode.

    Returns (unlearned_model, info). Pass `bn_mode` with `allow_bn_override=True` to run
    a method in the other mode.
    """
    spec = get(name)
    hp = dict(spec["hparams"])
    if overrides:
        unknown = set(overrides) - set(hp) - {"saliency_batches", "retain_fraction"}
        if unknown:
            raise KeyError(f"{name!r}: unknown hyperparameter override(s) {sorted(unknown)}")
        hp.update(overrides)

    declared = spec["bn_mode"]
    mode = declared if bn_mode is None else bn_mode
    overridden = mode != declared
    if overridden and not allow_bn_override:
        raise ValueError(
            f"{name!r} declares bn_mode={declared!r} but bn_mode={mode!r} was requested; "
            "pass allow_bn_override=True to run it anyway.")

    if not inplace:
        model = copy.deepcopy(model)
    model = model.to(device)

    if log:
        log(f"[run] method={name} bn_mode={mode} hparams={hp} seed={seed}")

    out_model, info = spec["fn"](model, loaders, device, bn_mode=mode, seed=seed,
                                 log=log, **hp)
    # bn_mode_ctx asserts the mode on exit and leaves the model in eval() for evaluation.
    if info.get("bn_mode") != mode:
        raise AssertionError(
            f"{name!r}: method reported bn_mode={info.get('bn_mode')!r}, expected {mode!r}")
    out_model.eval()

    info.update({"method": name, "bn_mode_declared": declared, "bn_mode_used": mode,
                 "bn_mode_overridden": overridden, "seed": seed})
    return out_model, info
