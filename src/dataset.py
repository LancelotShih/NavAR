"""ADE20K floor segmentation dataset: (image, mask) pairs from splits/*.txt."""

from pathlib import Path

import albumentations as A # needed for image augmentation to support segmentation classification
import cv2 # openCV for image processing tools
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2 # class converts numpy arrays into pytorch tensors
from torch.utils.data import Dataset

cv2.setNumThreads(0)

REPO = Path(__file__).resolve().parent.parent
IMAGES = REPO / "data/raw/ADEChallengeData2016/images"
MASKS = REPO / "data/ade20k_floor/masks"
# since the dataset has no test dataset, part of the training becomes validation instead
# This hashmap is just meant to map train.txt to the subset of images/training folder in the data etc
FOLDER = {"train": "training", "val": "training", "test": "validation"}
IGNORE = 255

def build_transform(train: bool, size: int = 256) -> A.Compose:
    resize = A.SmallestMaxSize(max_size=size)
    if train:
        ops = [
            resize,
            A.RandomCrop(height = size, width = size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5),
        ]
    else:
        ops = [
            resize,
            A.PadIfNeeded(
                min_height=None, min_width=None,
                pad_height_divisor=32, pad_width_divisor=32,
                position="top_left",
                border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=IGNORE,
            ),
        ]
    return A.Compose(ops + [A.Normalize(), ToTensorV2()])

class FloorDataset(Dataset):
    def __init__(self, split: str, transform: A.Compose | None = None):
        self.split = split
        self.ids = (REPO / "splits" / f"{split}.txt").read_text().split()
        self.folder = FOLDER[split]
        self.transform = transform or build_transform(train=(split == "train"))

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        id_ = self.ids[i]
        image = cv2.imread(str(IMAGES / self.folder / f"{id_}.jpg"), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(str(IMAGES / self.folder / f"{id_}.jpg"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(MASKS / self.folder / f"{id_}.png"), cv2.IMREAD_UNCHANGED)

        out = self.transform(image=image, mask=mask)
        return out["image"], out["mask"].long()