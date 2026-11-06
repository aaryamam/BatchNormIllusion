"""Tiny-ImageNet, read from `<DATA_ROOT>/tiny-imagenet-200` as distributed.

Returns the (train_aug, train_det, test_det) triple that `build_loaders` takes as `datasets`.
The test split is the labelled `val` directory. ResNet-50 runs at the native 64x64; ViT-S/16
upsamples to 224 and uses its pretrained normalisation.
"""

import os

import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from .cifar import DEFAULT_DATA_ROOT

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VIT_MEAN = (0.5, 0.5, 0.5)
VIT_STD = (0.5, 0.5, 0.5)


class TinyImageNetVal(Dataset):
    """The labelled validation split, with labels in the training set's class order."""

    def __init__(self, root, class_to_idx, transform):
        self.transform = transform
        self.samples, self.targets = [], []
        with open(os.path.join(root, "val", "val_annotations.txt")) as f:
            for line in f:
                name, wnid = line.split("\t")[:2]
                self.samples.append(os.path.join(root, "val", "images", name))
                self.targets.append(class_to_idx[wnid])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        with Image.open(self.samples[i]) as img:
            return self.transform(img.convert("RGB")), self.targets[i]


def tiny_datasets(arch="resnet50", data_root=DEFAULT_DATA_ROOT):
    root = os.path.join(data_root, "tiny-imagenet-200")
    if arch == "vit_s16":
        norm = T.Normalize(VIT_MEAN, VIT_STD)
        train_tf = T.Compose([T.Resize(224), T.RandomHorizontalFlip(), T.ToTensor(), norm])
        eval_tf = T.Compose([T.Resize(224), T.ToTensor(), norm])
    else:
        norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        train_tf = T.Compose([T.RandomCrop(64, padding=8), T.RandomHorizontalFlip(),
                              T.ToTensor(), norm])
        eval_tf = T.Compose([T.ToTensor(), norm])
    train = os.path.join(root, "train")
    tr_aug = torchvision.datasets.ImageFolder(train, transform=train_tf)
    tr_det = torchvision.datasets.ImageFolder(train, transform=eval_tf)
    return tr_aug, tr_det, TinyImageNetVal(root, tr_det.class_to_idx, eval_tf)
