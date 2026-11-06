"""Forget / retain accuracy on both the train and test splits, labelled separately.

Every evaluation forces `model.eval()`: inference in train mode would overwrite the BN
running buffers and produce meaningless accuracy on single-class loaders.
"""

import torch


@torch.no_grad()
def accuracy(model, loader, device):
    """Top-1 accuracy in percent. Returns NaN for an empty/None loader."""
    if loader is None:
        return float("nan")
    was_training = model.training
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    model.train(was_training)
    return 100.0 * correct / total if total else float("nan")


def evaluate_all(model, loaders, device):
    """The full accuracy panel, both splits, explicitly labelled."""
    return {
        "forget_train": accuracy(model, loaders["forget_eval_train"], device),
        "retain_train": accuracy(model, loaders["retain_eval_train"], device),
        "forget_test": accuracy(model, loaders["forget_eval_test"], device),
        "retain_test": accuracy(model, loaders["retain_eval_test"], device),
        "test_full": accuracy(model, loaders["test_full"], device),
    }
