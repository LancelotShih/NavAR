# a floor detector that doesn't use any learning

import argparse
import csv
import json
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from src.dataset import FloorDataset
from src.metrics import FloorMetrics

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "runs" / "baseline"
THRESHOLDS = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 25, 30, 40, 50, 75, 100]

def baseline_transform(size: int = 256) -> A.Compose:
    # resizing photo so the short side is 256, same as UNet sizing
    return A.Compose([A.SmallestMaxSize(max_size = size), ToTensorV2()])

def seed_patch(h: int, w: int) -> tuple[slice, slice]:
    # assumes bottom 20% of rows and middle 30% of columns are always going to be floor
    return slice(int(0.8 * h), h), slice(int(0.35 * w), int(0.65 * w))

def color_distance(rgb: np.ndarray) -> np.ndarray:
    # how different is the current pixel's color from the "floor" colors in the box?
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    rows, cols = seed_patch(h, w)
    seed = lab[rows, cols].reshape(-1, 3)

    mean = seed.mean(axis=0)
    cov = np.cov(seed, rowvar=False) + np.eye(3)
    inv = np.linalg.inv(cov)

    diff = lab.reshape(-1, 3) - mean
    d2 = np.einsum("ij,jk,ik->i", diff, inv, diff)# 
    return d2.reshape(h, w)

def to_rgb(image: torch.Tensor) -> np.ndarray:
    # convert between dataset color format to OpenCV color format to perform color distance
    return np.ascontiguousarray(image.permute(1, 2, 0).numpy())

def tune(split: str = "val", workers: int = 6) -> None:
    # decides the threshold for what distance is considered "floor"
    ds = FloorDataset(split, transform=baseline_transform())
    loader = DataLoader(ds, batch_size=None, num_workers=workers)
    metrics = {t: FloorMetrics() for t in THRESHOLDS}

    for image, mask in loader:
        d2 = torch.from_numpy(color_distance(to_rgb(image)))
        for t, m in metrics.items():
            m.update(d2 < t, mask) # the pixel is "floor" if distance "d2" is less than "t"

    rows = [{"threshold": t, **m.compute()} for t, m in metrics.items()]
    best = max(rows, key=lambda r: r["iou"])

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"threshold_sweep_{split}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (OUT / "threshold.json").write_text(json.dumps(best, indent=2))

    for r in rows:
        mark = "  ← best" if r is best else ""
        print(f"threshold {r['threshold']:>5}  IoU {r['iou']:.4f}  "
              f"P {r['precision']:.3f}  R {r['recall']:.3f}{mark}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    tune(args.split, args.workers)
