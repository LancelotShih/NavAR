"""
Download and prepare ADE20K dataset and marge labels into floor, not floor, or ignore classes

Allows user to define the output layout as well 
"""

import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

URL = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
SHA256 = "7ff1be44964418441f542a7cc1e1a650e7dc0fc275f5d23252bc9bbdbc977b29"
ROOT_NAME = "ADEChallengeData2016"
SPLITS = ("training", "validation")
REPO = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(zip_path: Path) -> None:
    if zip_path.exists():
        print(f"found {zip_path}")
    else:
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = zip_path.with_suffix(".part")
        with urllib.request.urlopen(URL) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            with tqdm(total=total, unit="B", unit_scale=True, desc="download") as bar:
                for chunk in iter(lambda: r.read(1 << 20), b""):
                    f.write(chunk)
                    bar.update(len(chunk))
        tmp.rename(zip_path)

    digest = sha256(zip_path)
    if digest != SHA256:
        sys.exit(f"checksum mismatch for {zip_path}: {digest} (expected {SHA256})")


def extract(zip_path: Path, raw_dir: Path) -> Path:
    root = raw_dir / ROOT_NAME
    if (root / "objectInfo150.txt").exists():
        print(f"found {root}")
        return root
    print(f"extracting {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(raw_dir)
    return root


def load_object_info(path: Path) -> dict[int, str]:
    """Return {class index (1-150): Name} from objectInfo150.txt."""
    lines = path.read_text().splitlines()
    header = lines[0].split("\t")
    idx_col, name_col = header.index("Idx"), header.index("Name")
    info = {}
    for line in lines[1:]:
        if line.strip():
            fields = line.split("\t")
            info[int(fields[idx_col])] = fields[name_col].strip()
    return info


def resolve(names: list[str], info: dict[int, str]) -> dict[str, int]:
    """Map each configured name to the single class index it matches."""
    resolved = {}
    for name in names:
        key = name.strip().lower()
        hits = [
            idx for idx, full in info.items()
            if key == full.lower() or key in (s.strip().lower() for s in full.split(","))
        ]
        if len(hits) != 1:
            found = [info[i] for i in hits] or "nothing"
            sys.exit(f"label name {name!r} must match exactly one class, matched {found}")
        resolved[name] = hits[0]
    return resolved


def build_lut(info, floor_ids, ignore_ids, values) -> np.ndarray:
    lut = np.full(256, values["ignore"], dtype=np.uint8)  # 0 = unlabeled, >150 unused
    lut[list(info)] = values["not_floor"]
    lut[floor_ids] = values["floor"]
    lut[ignore_ids] = values["ignore"]
    return lut


def convert(job):
    src, dst, lut, overwrite = job
    if overwrite or not dst.exists():
        ann = np.array(Image.open(src))
        if ann.ndim != 2 or ann.dtype != np.uint8:
            raise ValueError(f"unexpected annotation format in {src}: {ann.shape} {ann.dtype}")
        Image.fromarray(lut[ann]).save(dst)
    mask = np.array(Image.open(dst))
    return np.bincount(mask.ravel(), minlength=256)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, default=REPO / "data/raw", help="download/extract directory")
    p.add_argument("--out", type=Path, default=REPO / "data/ade20k_floor", help="merged mask cache")
    p.add_argument("--labels", type=Path, default=REPO / "configs/labels.yaml")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--overwrite", action="store_true", help="regenerate existing masks")
    args = p.parse_args()

    zip_path = args.raw / f"{ROOT_NAME}.zip"
    download(zip_path)
    root = extract(zip_path, args.raw)

    cfg = yaml.safe_load(args.labels.read_text())
    values = cfg["output_values"]
    info = load_object_info(root / "objectInfo150.txt")
    floor = resolve(cfg["floor"], info)
    ignore = resolve(cfg["ignore"], info)
    overlap = set(floor.values()) & set(ignore.values())
    if overlap:
        sys.exit(f"classes listed as both floor and ignore: {[info[i] for i in overlap]}")

    print("floor:  " + ", ".join(f"{i} {info[i]!r}" for i in floor.values()))
    print("ignore: 0 (unlabeled), " + ", ".join(f"{i} {info[i]!r}" for i in ignore.values()))
    lut = build_lut(info, list(floor.values()), list(ignore.values()), values)

    stats = {}
    for split in SPLITS:
        src_dir = root / "annotations" / split
        dst_dir = args.out / "masks" / split
        dst_dir.mkdir(parents=True, exist_ok=True)
        jobs = [(f, dst_dir / f.name, lut, args.overwrite) for f in sorted(src_dir.glob("*.png"))]
        counts = np.zeros(256, dtype=np.int64)
        with Pool(args.workers) as pool:
            for c in tqdm(pool.imap_unordered(convert, jobs, chunksize=64), total=len(jobs), desc=split):
                counts += c
        stats[split] = {
            "masks": len(jobs),
            "pixels": {k: int(counts[v]) for k, v in values.items()},
        }
        labelled = counts[values["floor"]] + counts[values["not_floor"]]
        print(f"{split}: {len(jobs)} masks, floor = {counts[values['floor']] / labelled:.1%} of labelled pixels, "
              f"ignore = {counts[values['ignore']] / counts.sum():.1%} of all pixels")

    meta = {
        "source": URL,
        "images": {s: str(root / "images" / s) for s in SPLITS},
        "values": values,
        "floor_classes": {info[i]: i for i in floor.values()},
        "ignore_classes": {"unlabeled": 0, **{info[i]: i for i in ignore.values()}},
        "stats": stats,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {args.out / 'meta.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
