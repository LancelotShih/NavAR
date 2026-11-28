"""Turn LabelMe annotations of my own photos into floor masks (0 = not floor, 1 = floor, 255 = ignore).

Usage:  python scripts/labelme_to_masks.py [batch]          (default batch: self_batch_2)
Reads   data/<batch>/*.jpg  and  data/labels/<batch>/*.json   (written by scripts/label_server.sh)
Writes  data/self_floor/masks/<batch>/<photo>.png            same size as the upright photo
        data/self_floor/preview/<batch>/<photo>.jpg          small overlay to check by eye
        splits/<batch>.txt                                   photo names, one per line

Rules (same as configs/labels.yaml for ADE20K):
  "floor" shapes -> 1; "not_floor" shapes -> 0, drawn over floor to cut out things standing on it;
  "ignore" shapes -> 255, drawn last so they win; everything not covered by a shape -> 0 (not floor).
Every photo must have a label file. A photo with no floor at all needs the "no_floor" flag ticked,
so an unlabelled photo is never silently counted as floor-free; the "exclude" flag leaves a photo out.
"""

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

REPO = Path(__file__).resolve().parent.parent
VALUES = {"floor": 1, "not_floor": 0, "ignore": 255}
DRAW_ORDER = ["floor", "not_floor", "ignore"]  # later labels overwrite earlier ones


def shape_mask(shape: dict, h: int, w: int) -> np.ndarray:
    # rasterise one LabelMe shape into an (h, w) boolean mask
    kind, pts = shape["shape_type"], shape["points"]
    if kind == "mask":
        # AI outline stored as a small PNG covering the box between its two points
        small = np.array(Image.open(io.BytesIO(base64.b64decode(shape["mask"])))).astype(bool)
        if small.ndim == 3:
            small = small[..., 0]
        x0, y0 = (int(round(v)) for v in pts[0])
        out = np.zeros((h, w), bool)
        ys, xs = slice(max(y0, 0), min(y0 + small.shape[0], h)), slice(max(x0, 0), min(x0 + small.shape[1], w))
        out[ys, xs] = small[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0]
        return out
    canvas = Image.new("1", (w, h), 0)
    draw = ImageDraw.Draw(canvas)
    if kind == "polygon":
        if len(pts) < 3:
            raise ValueError("polygon with fewer than 3 points")
        draw.polygon([tuple(p) for p in pts], outline=1, fill=1)
    elif kind == "rectangle":
        (x0, y0), (x1, y1) = pts
        draw.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], outline=1, fill=1)
    else:
        raise ValueError(f"shape type {kind!r} can't mark an area; use polygon, rectangle or AI outline")
    return np.array(canvas, bool)


def convert(photo: Path, label_file: Path) -> tuple[np.ndarray | None, str]:
    # returns (mask, note); mask is None for an excluded photo
    ann = json.loads(label_file.read_text())
    flags = {k for k, v in (ann.get("flags") or {}).items() if v}
    if "exclude" in flags:
        return None, "excluded"

    with Image.open(photo) as im:
        w, h = ImageOps.exif_transpose(im).size      # upright size, as LabelMe and cv2.imread see it
    if (ann["imageWidth"], ann["imageHeight"]) != (w, h):
        raise ValueError(f"label is {ann['imageWidth']}x{ann['imageHeight']} but photo is {w}x{h}")

    shapes = ann.get("shapes") or []
    unknown = sorted({s["label"] for s in shapes} - set(VALUES))
    if unknown:
        raise ValueError(f"unknown labels {unknown}; only {list(VALUES)} are allowed")
    has_floor = any(s["label"] == "floor" for s in shapes)
    if not has_floor and "no_floor" not in flags:
        raise ValueError("no floor shapes; tick the no_floor flag if the photo really has no floor")
    if has_floor and "no_floor" in flags:
        raise ValueError("ticked no_floor but has floor shapes")

    mask = np.zeros((h, w), np.uint8)
    for label in DRAW_ORDER:
        for s in shapes:
            if s["label"] == label:
                mask[shape_mask(s, h, w)] = VALUES[label]
    return mask, ""


def preview(photo: Path, mask: np.ndarray, path: Path) -> None:
    # small overlay: green = floor, red = ignore
    with Image.open(photo) as im:
        rgb = np.array(ImageOps.exif_transpose(im).convert("RGB"), np.float32)
    for value, colour in ((1, (0, 255, 0)), (255, (255, 0, 0))):
        sel = mask == value
        rgb[sel] = 0.55 * rgb[sel] + 0.45 * np.array(colour)
    out = Image.fromarray(rgb.astype(np.uint8))
    out.thumbnail((800, 800))
    out.save(path, quality=85)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("batch", nargs="?", default="self_batch_2")
    args = ap.parse_args()

    photos = sorted(p for p in (REPO / "data" / args.batch).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    labels = REPO / "data" / "labels" / args.batch
    masks_dir = REPO / "data" / "self_floor" / "masks" / args.batch
    preview_dir = REPO / "data" / "self_floor" / "preview" / args.batch
    masks_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    kept, problems, excluded = [], [], []
    counts = np.zeros(3, np.int64)  # floor, not floor, ignore pixels
    no_floor = 0
    for photo in photos:
        label_file = labels / f"{photo.stem}.json"
        if not label_file.exists():
            problems.append(f"{photo.name}: not labelled yet")
            continue
        try:
            mask, note = convert(photo, label_file)
        except (ValueError, KeyError) as e:
            problems.append(f"{photo.name}: {e}")
            continue
        if mask is None:
            excluded.append(photo.name)
            continue
        Image.fromarray(mask).save(masks_dir / f"{photo.stem}.png")
        preview(photo, mask, preview_dir / f"{photo.stem}.jpg")
        counts += [(mask == 1).sum(), (mask == 0).sum(), (mask == 255).sum()]
        no_floor += not (mask == 1).any()
        kept.append(photo.stem)

    (REPO / "splits" / f"{args.batch}.txt").write_text("".join(f"{s}\n" for s in kept))
    share = counts / max(counts.sum(), 1)
    print(f"{len(kept)} masks written to {masks_dir.relative_to(REPO)} "
          f"({no_floor} with no floor); {len(excluded)} excluded")
    print(f"pixels: floor {share[0]:.1%}, not floor {share[1]:.1%}, ignore {share[2]:.1%}")
    print(f"check overlays in {preview_dir.relative_to(REPO)}  (green = floor, red = ignore)")
    if problems:
        print(f"\n{len(problems)} photo(s) need attention:", *problems, sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
