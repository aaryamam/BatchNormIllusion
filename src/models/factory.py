"""Architectures, built by name.

`resnet18`     - CIFAR ResNet-18 with BatchNorm
`resnet18_gn`  - the same network with GroupNorm(32, c)
`resnet50`     - ResNet-50 with the same 3x3 stem and no max-pool, for 64x64 Tiny-ImageNet
`vit_s16`      - ViT-S/16 at 224 (LayerNorm), optionally ImageNet-pretrained
"""

from functools import partial

import torch.nn as nn
import torchvision

from .resnet_cifar import resnet18_cifar

VIT_NAME = "vit_small_patch16_224.augreg_in21k_ft_in1k"


def resnet50_small_stem(num_classes=200):
    model = torchvision.models.resnet50(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def vit_s16(num_classes=200, pretrained=False):
    import timm

    return timm.create_model(VIT_NAME, pretrained=pretrained, num_classes=num_classes)


ARCHS = {
    "resnet18": partial(resnet18_cifar, norm="bn"),
    "resnet18_gn": partial(resnet18_cifar, norm="gn"),
    "resnet50": resnet50_small_stem,
    "vit_s16": vit_s16,
}


def build_model(arch, num_classes, pretrained=False):
    kwargs = {"pretrained": True} if pretrained else {}
    model = ARCHS[arch](num_classes=num_classes, **kwargs)
    # Lets the teacher-based methods draw a random network of the same architecture.
    model.fresh_copy = partial(build_model, arch, num_classes)
    return model
