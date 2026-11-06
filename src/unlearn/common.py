"""Shared machinery for the unlearning methods, including the BN-mode contract.

`bn_mode_ctx` sets the declared BN mode, asserts it, and re-asserts it on exit.
"""

import contextlib

import torch
import torch.nn as nn

from ..models.resnet_cifar import bn_layers, resnet18_cifar


def set_bn_mode(model, mode):
    """Put ONLY the normalisation layers into `mode`; everything else trains.

    'train' - BN uses batch statistics and UPDATES its running buffers.
    'eval'  - BN uses its frozen running buffers and does not update them.
    """
    if mode not in ("train", "eval"):
        raise ValueError(f"bn_mode must be 'train' or 'eval', got {mode!r}")
    model.train()  # everything trainable
    if mode == "eval":
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()
    return model


def assert_bn_mode(model, mode):
    """Hard assertion that every BN module is in the declared mode."""
    bns = bn_layers(model)
    if not bns:
        return  # GroupNorm variant
    want_training = (mode == "train")
    bad = [n for n, m in bns if m.training != want_training]
    if bad:
        raise AssertionError(
            f"BN mode contract violated: expected bn_mode={mode!r} "
            f"(module.training={want_training}) but {len(bad)} layer(s) differ, "
            f"e.g. {bad[:3]}")


@contextlib.contextmanager
def bn_mode_ctx(model, mode, log=None):
    """Enter the declared BN mode, assert it, and assert it still holds on exit.

    On a clean exit the model is left in `eval()` — the correct state for the
    evaluation that always follows unlearning. Callers must therefore NOT re-assert
    the unlearning-time BN mode after this context closes; the exit assertion inside
    the `finally` block is the authoritative check that the mode held for the whole
    loop, and it runs before the model is put into eval().
    """
    set_bn_mode(model, mode)
    assert_bn_mode(model, mode)
    if log is not None:
        log(f"[bn] entering unlearning with bn_mode={mode!r} "
            f"({len(bn_layers(model))} BN layers)")
    ok = False
    try:
        yield model
        ok = True
    finally:
        if ok:
            # nothing in the loop may have flipped the mode
            assert_bn_mode(model, mode)
            model.eval()


def head_size(model):
    """Number of output classes, for ResNets (`fc`) and timm ViTs (`num_classes`)."""
    return model.fc.out_features if hasattr(model, "fc") else model.num_classes


def random_init_like(model):
    """A freshly initialised network of the same architecture (incompetent teacher)."""
    factory = getattr(model, "fresh_copy", None)
    return factory() if factory else resnet18_cifar(num_classes=head_size(model), norm="bn")


class StepBudget:
    """Iterate a loader indefinitely for exactly `steps` batches."""

    def __init__(self, loader, steps):
        self.loader = loader
        self.steps = steps

    def __iter__(self):
        it = iter(self.loader)
        for i in range(self.steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(self.loader)
                batch = next(it)
            yield i + 1, batch


def sgd_ascent_step(model, loss, lr):
    """One plain gradient-ascent step: theta <- theta + lr * grad."""
    model.zero_grad(set_to_none=True)
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.data.add_(p.grad, alpha=lr)


def sgd_descent_step(model, loss, lr):
    """One plain gradient-DESCENT step: theta <- theta - lr * grad."""
    model.zero_grad(set_to_none=True)
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.data.add_(p.grad, alpha=-lr)
