"""Run a trained floor model on your own images and save overlays to look at.

Usage (from the repo root):
    python -m src.predict photo.jpg other.png          # one or more images
    python -m src.predict path/to/folder/              # every image in a folder
    python -m src.predict ADE_val_00000009             # an ADE20K id (also shows the label)

Each input gets one PNG in outputs/predict/: the photo, the predicted floor in green,
and, for ADE20K images, the ground-truth mask (green = floor, red = ignore).
By default the checkpoint with the best val IoU under runs/ is used; pick another with --ckpt.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.dataset import IMAGES, MASKS, build_transform
from src.model import build_model

REPO = Path(__file__).resolve().parent.parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def best_checkpoint() -> Path:
    """The runs/*/best.pt with the highest val IoU recorded inside it."""
    best, best_iou = None, -1.0
    for path in sorted((REPO / "runs").glob("*/best.pt")):
        iou = torch.load(path, map_location="cpu", weights_only=False)["val"]["iou"]
        if iou > best_iou:
            best, best_iou = path, iou
    if best is None:
        raise SystemExit("no runs/*/best.pt found; train a model first or pass --ckpt")
    return best


def resolve_inputs(items: list[str]) -> list[Path]:
    """Expand folders and ADE20K ids into image paths."""
    paths = []
    for item in items:
        p = Path(item)
        if p.is_dir():
            paths += sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXTS)
        elif p.exists():
            paths.append(p)
        else:
            hits = list(IMAGES.glob(f"*/{item}.jpg"))
            if not hits:
                raise SystemExit(f"not a file, folder or ADE20K id: {item}")
            paths.append(hits[0])
    return paths


def ade_label(path: Path) -> np.ndarray | None:
    """The merged 0/1/255 mask if this image is part of ADE20K, else None."""
    mask_path = MASKS / path.parent.name / f"{path.stem}.png"
    if path.parent.parent == IMAGES and mask_path.exists():
        return cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    return None


def tint(image: np.ndarray, region: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = image.copy()
    out[region] = (0.5 * out[region] + 0.5 * np.array(color)).astype(np.uint8)
    return out


@torch.no_grad()
def predict(model, transform, image_rgb: np.ndarray, device) -> np.ndarray:
    """Floor probability (H, W) at the image's original resolution."""
    h, w = image_rgb.shape[:2]
    x = transform(image=image_rgb)["image"][None].to(device)
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        logits = model(x)
    # undo the padding (it was added on the bottom/right), then scale back up to full size
    s = 256 / min(h, w)
    rh, rw = round(h * s), round(w * s)
    logits = logits[:, :, :rh, :rw].float()
    logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
    return torch.sigmoid(logits)[0, 0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="image files, folders, or ADE20K ids")
    ap.add_argument("--ckpt", type=Path, help="checkpoint (default: best val IoU under runs/)")
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "predict")
    ap.add_argument("--threshold", type=float, default=0.5, help="floor if probability > this")
    ap.add_argument("--cpu", action="store_true", help="run on CPU even if a GPU is available")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    ckpt_path = args.ckpt or best_checkpoint()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(pretrained=False).to(device).eval()
    model.load_state_dict(ckpt["model"])
    print(f"model: {ckpt_path} (epoch {ckpt['epoch']}, val IoU {ckpt['val']['iou']:.4f}) on {device}")

    transform = build_transform(train=False)
    args.out.mkdir(parents=True, exist_ok=True)

    for path in resolve_inputs(args.inputs):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"  skip {path}: could not read image")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        prob = predict(model, transform, rgb, device)
        floor = prob > args.threshold

        panels = [rgb, tint(rgb, floor, (0, 255, 0))]
        note = ""
        label = ade_label(path)
        if label is not None:
            panels.append(tint(tint(rgb, label == 1, (0, 255, 0)), label == 255, (255, 0, 0)))
            valid = label != 255
            truth = label == 1
            inter = (floor & truth & valid).sum()
            union = ((floor | truth) & valid).sum()
            note = f", IoU vs label {inter / union:.3f}" if union else ", no floor in label or prediction"

        out_path = args.out / f"{path.stem}_floor.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(np.hstack(panels), cv2.COLOR_RGB2BGR))
        print(f"  {path.name}: {floor.mean():.1%} floor{note} -> {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
