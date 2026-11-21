# baseline and trained model scored on validation images according to same metrics as metrics.py
# evaluate.py then judges whether the model did better or the baseline did better

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg") # draw to files; no screen needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.baseline import OUT as BASELINE_DIR, baseline_transform, color_distance, to_rgb
from src.dataset import FOLDER, IGNORE, FloorDataset
from src.metrics import FloorMetrics
from src.model import build_model
from src.predict import best_checkpoint, tint

REPO = Path(__file__).resolve().parent.parent
ADE = REPO / "data/raw/ADEChallengeData2016"
META = json.loads((REPO / "data/ade20k_floor/meta.json").read_text())
FLOOR_CLASSES = {name.split(",")[0]: idx for name, idx in META["floor_classes"].items()}
CLASS_NAMES = {int(l.split("\t")[0]): l.split("\t")[-1].split(",")[0].strip()
               for l in (ADE / "objectInfo150.txt").read_text().splitlines()[1:] if l.strip()}
METHODS = ("baseline", "unet")


def ade_classes(split: str, id_: str, shape: tuple[int, int]) -> np.ndarray:
    # need to pull original ADE dataset classes to answer "was this false alarm on a road or a wall"
    a = cv2.imread(str(ADE / "annotations" / FOLDER[split] / f"{id_}.png"), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise FileNotFoundError(str(ADE / "annotations" / FOLDER[split] / f"{id_}.png"))
    return cv2.resize(a, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def image_iou(pred: torch.Tensor, mask: torch.Tensor) -> float | None:
    # gives us a per image intersection over union results, since FloorMetrics gives it for the whole split
    valid, truth = mask != IGNORE, mask == 1
    union = ((pred | truth) & valid).sum().item()
    return ((pred & truth & valid).sum().item() / union) if union else None


@torch.no_grad()
def predict_both(model, threshold, x, image, h, w, device) -> dict[str, torch.Tensor]:
    # calls the unet model and baseline model to do a prediction on an image
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(x[None].to(device))
    return {
        "unet": (logits[0, 0, :h, :w] > 0).cpu(), # crop off padding
        "baseline": torch.from_numpy(color_distance(to_rgb(image)) < threshold),
    }

def evaluate(split, model, threshold, device, workers):
    # runs "predict_both" on every image then for each tallies up the valid pixels predicted 
    model_ds = FloorDataset(split) # normalised + padded, for the U-Net
    base_ds = FloorDataset(split, transform=baseline_transform()) # raw RGB, unpadded
    pairs = zip(DataLoader(model_ds, batch_size=None, num_workers=workers // 2),
                DataLoader(base_ds, batch_size=None, num_workers=workers // 2))

    metrics = {m: FloorMetrics() for m in METHODS}
    said_floor = {m: np.zeros(151, np.int64) for m in METHODS} # predicted floor, by true class
    total = np.zeros(151, np.int64) # valid pixels, by true class
    per_image = []

    for i, ((x, _), (image, mask)) in enumerate(tqdm(pairs, total=len(model_ds), desc=split)):
        h, w = mask.shape
        preds = predict_both(model, threshold, x, image, h, w, device)
        cls = ade_classes(split, model_ds.ids[i], (h, w))
        valid = (mask != IGNORE).numpy()
        total += np.bincount(cls[valid], minlength=151)

        row = {"id": model_ds.ids[i], "floor_frac": (mask == 1).float().mean().item()}
        for m, p in preds.items():
            metrics[m].update(p, mask)
            said_floor[m] += np.bincount(cls[valid & p.numpy()], minlength=151)
            row[m] = image_iou(p, mask)
            row[f"{m}_pred_frac"] = p.float().mean().item()
        per_image.append(row)

    return metrics, said_floor, total, per_image

def summarize(metrics, said_floor, total, per_image):
    # turns the tallies of valid pixels for each model into an output we can read
    floor_ids = set(FLOOR_CLASSES.values())
    no_floor = [r for r in per_image if r["floor_frac"] == 0]
    out = {}
    for m in METHODS:
        fp = said_floor[m].copy()
        fp[list(floor_ids)] = 0
        top_fp = np.argsort(-fp)[:6]
        out[m] = {
            **metrics[m].compute(),
            "recall_by_class": {n: said_floor[m][c] / max(total[c], 1) for n, c in FLOOR_CLASSES.items()},
            "false_alarms_by_class": {CLASS_NAMES[c]: {"share_of_fp": fp[c] / max(fp.sum(), 1),
                                                       "rate": said_floor[m][c] / max(total[c], 1)}
                                      for c in top_fp},
            "no_floor_images_with_false_floor": sum(r[f"{m}_pred_frac"] > 0.01 for r in no_floor) / len(no_floor),
        }
    return out


def pick_examples(per_image) -> list[str]:
    # grabs a few examples from the 90, 50 and 10th percentile of performance for both models to showcase results
    ranked = sorted((r for r in per_image if r["floor_frac"] > 0.05 and r["unet"] is not None),
                    key=lambda r: r["unet"])
    return [ranked[int(q * (len(ranked) - 1))]["id"] for q in (0.9, 0.5, 0.1)]

def save_figure(ids, split, model, threshold, device, path):
    # saves the examples for the report
    model_ds = FloorDataset(split)
    base_ds = FloorDataset(split, transform=baseline_transform())
    index = {id_: i for i, id_ in enumerate(model_ds.ids)}

    fig, axes = plt.subplots(len(ids), 4, figsize=(12, 2.6 * len(ids)), squeeze=False)
    for row, id_ in zip(axes, ids):
        x, _ = model_ds[index[id_]]
        image, mask = base_ds[index[id_]]
        h, w = mask.shape
        preds = predict_both(model, threshold, x, image, h, w, device)
        rgb = to_rgb(image)
        m = mask.numpy()
        panels = [
            (rgb, id_),
            (tint(tint(rgb, m == 1, (0, 255, 0)), m == IGNORE, (255, 0, 0)), "label"),
            (tint(rgb, preds["baseline"].numpy(), (0, 255, 0)), f"baseline  IoU {image_iou(preds['baseline'], mask) or 0:.2f}"),
            (tint(rgb, preds["unet"].numpy(), (0, 255, 0)), f"U-Net  IoU {image_iou(preds['unet'], mask) or 0:.2f}"),
        ]
        for ax, (img, title) in zip(row, panels):
            ax.imshow(img)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--ckpt", type=Path, help="default: best val IoU under runs/")
    ap.add_argument("--examples", nargs="*", default=[], help="extra ADE ids for the figure")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    device = torch.device("cuda")
    ckpt_path = args.ckpt or best_checkpoint()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(pretrained=False).to(device).eval()
    model.load_state_dict(ckpt["model"])
    threshold = json.loads((BASELINE_DIR / "threshold.json").read_text())["threshold"]
    print(f"U-Net: {ckpt_path}  |  baseline threshold: {threshold}")

    out = REPO / "outputs" / f"eval_{args.split}"
    out.mkdir(parents=True, exist_ok=True)

    metrics, said_floor, total, per_image = evaluate(args.split, model, threshold, device, args.workers)
    results = summarize(metrics, said_floor, total, per_image)
    results["checkpoint"], results["baseline_threshold"] = str(ckpt_path), threshold
    (out / "results.json").write_text(json.dumps(results, indent=2))
    with open(out / "per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
        w.writeheader()
        w.writerows(per_image)

    print(f"\n{'method':<10}{'IoU':>8}{'P':>8}{'R':>8}{'no-floor FA':>13}")
    for m in METHODS:
        r = results[m]
        print(f"{m:<10}{r['iou']:>8.4f}{r['precision']:>8.3f}{r['recall']:>8.3f}"
              f"{r['no_floor_images_with_false_floor']:>13.1%}")
    for m in METHODS:
        print(f"\n{m} recall by class: " +
              ", ".join(f"{n} {v:.1%}" for n, v in results[m]["recall_by_class"].items()))

    ids = pick_examples(per_image) + args.examples
    save_figure(ids, args.split, model, threshold, device, out / "qualitative.pdf")
    print(f"\nwrote {out}/results.json, per_image.csv, qualitative.pdf")


if __name__ == "__main__":
    main()
