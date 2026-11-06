"""RandomLabeling: a purely data-based baseline.

Each forget sample gets a uniformly random wrong label, then the model is fine-tuned by
gradient descent on those samples.
"""

import torch
import torch.nn as nn

from .common import StepBudget, assert_bn_mode, bn_mode_ctx, sgd_descent_step, head_size

BN_MODE = "train"
DEFAULTS = {"lr": 1e-3, "steps": 500}


def random_wrong_labels(y, num_classes=10, generator=None):
    """Uniform over the num_classes-1 labels excluding the true one.

    Implemented by drawing an offset in [1, num_classes-1] and adding it modulo
    num_classes, which is exactly uniform over the wrong labels and never collides
    with the true label.
    """
    offset = torch.randint(1, num_classes, y.shape, generator=generator,
                           device=y.device if generator is None else None)
    return (y + offset.to(y.device)) % num_classes


def unlearn_random_labeling(model, loaders, device, lr=1e-3, steps=500,
                            bn_mode=BN_MODE, seed=0, log=None, **_):
    num_classes = head_size(model)
    crit = nn.CrossEntropyLoss()
    torch.manual_seed(seed + 777)
    history = []
    with bn_mode_ctx(model, bn_mode, log=log):
        for step, (x, y) in StepBudget(loaders["forget_train_aug"], steps):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_rand = random_wrong_labels(y, num_classes)
            assert (y_rand != y).all(), "random label collided with the true label"
            loss = crit(model(x), y_rand)
            sgd_descent_step(model, loss, lr)   # DESCENT on the wrong labels
            assert_bn_mode(model, bn_mode)
            if log and (step % 100 == 0 or step == 1):
                log(f"  [RandomLabeling/{bn_mode}] step {step}/{steps} "
                    f"CE(random)={loss.item():.4f}")
            history.append({"step": step, "ce_random": float(loss.item())})

    return model, {"history": history, "bn_mode": bn_mode,
                   "hparams": {"lr": lr, "steps": steps, "num_classes": num_classes}}
