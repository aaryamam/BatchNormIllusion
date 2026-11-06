"""ResNet-18, CIFAR adaptation: 3x3 stem conv, no max-pool, 512 -> num_classes head.

Built on torchvision's `resnet18`, giving 20 BatchNorm2d layers. The GroupNorm variant
replaces every nn.BatchNorm2d(c) with nn.GroupNorm(32, c).
"""

import torch
import torch.nn as nn
import torchvision

NUM_BN_LAYERS = 20
FEATURE_DIM = 512


def _to_groupnorm(module, groups=32):
    """Recursively replace every BatchNorm2d(c) with GroupNorm(groups, c)."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(groups, child.num_features))
        else:
            _to_groupnorm(child, groups)
    return module


def resnet18_cifar(num_classes=10, norm="bn", gn_groups=32):
    """CIFAR ResNet-18.

    Args:
        num_classes: size of the final FC.
        norm: 'bn' (default) or 'gn'.
        gn_groups: groups for the GroupNorm variant.
    """
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    # CIFAR stem: 3x3 stride 1 pad 1, and drop the max-pool.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(FEATURE_DIM, num_classes)

    if norm == "gn":
        model = _to_groupnorm(model, gn_groups)
    elif norm != "bn":
        raise ValueError(f"norm must be 'bn' or 'gn', got {norm!r}")

    if norm == "bn":
        n = len(bn_layers(model))
        assert n == NUM_BN_LAYERS, f"expected {NUM_BN_LAYERS} BN layers, found {n}"
    return model


def bn_layers(model):
    """Ordered [(name, module)] of every BatchNorm2d, in forward-execution order.

    `named_modules()` walks the module tree in registration order, which for this
    architecture coincides with execution order except that a block's `downsample`
    BN is registered after the block's bn1/bn2. That is the natural grouping for a
    depth profile, so we keep registration order and record the index explicitly.
    """
    return [(n, m) for n, m in model.named_modules() if isinstance(m, nn.BatchNorm2d)]


@torch.no_grad()
def bn_state(model):
    """Snapshot {layer_name: (running_mean, running_var)} as detached CPU clones."""
    return {
        n: (m.running_mean.detach().cpu().clone(), m.running_var.detach().cpu().clone())
        for n, m in bn_layers(model)
    }


class FeatureExtractor(nn.Module):
    """Wraps the model to return penultimate 512-d features (post-avgpool)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        m = self.model
        out = m.relu(m.bn1(m.conv1(x)))
        out = m.maxpool(out)
        out = m.layer1(out)
        out = m.layer2(out)
        out = m.layer3(out)
        out = m.layer4(out)
        out = m.avgpool(out)
        return torch.flatten(out, 1)


@torch.no_grad()
def extract_features(model, loader, device, max_batches=None):
    """Return (features [N,512], labels [N]) with the model in eval mode.

    eval mode matters: a forward pass in train mode would overwrite the very BN
    running statistics this project is measuring.
    """
    was_training = model.training
    model.eval()
    fx = FeatureExtractor(model)
    feats, labs = [], []
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        feats.append(fx(x.to(device, non_blocking=True)).cpu())
        labs.append(y.cpu())
    model.train(was_training)
    return torch.cat(feats), torch.cat(labs)
