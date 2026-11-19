"""Write fixed train/val/test ID lists for ADE20K floor segmentation."""

import argparse
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# we need to sort the IDs that have both a merged mask and an image
def ids_in(split: str, masks: Path, images: Path) -> list[str]:
    ids = [p.stem for p in (masks / split).glob("*.png")]
    ids = [i for i in ids if (images / split / f"{i}.jpg").exists()]
    return sorted(ids)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type = float, default = 0.1)
    ap.add_argument("--seed", type = int, default = 0)
    ap.add_argument("--masks", type = Path, default = REPO / "data/ade20k_floor/masks")
    ap.add_argument("--images", type = Path, default = REPO / "data/raw/ADEChallengeData2016/images")
    ap.add_argument("--out", type = Path, default = REPO / "splits")
    args = ap.parse_args()

    pool = ids_in("training", args.masks, args.images)
    test = ids_in("validation", args.masks, args.images)

    rng = random.Random(args.seed)
    rng.shuffle(pool)
    n_val = round(len(pool) * args.val_frac)
    val, train = sorted(pool[:n_val]), sorted(pool[n_val:])

    assert not set(train) & set(val), "train/val have overlap"

    args.out.mkdir(exist_ok=True)
    for name, ids in (("train", train), ("val", val), ("test", test)):
        (args.out / f"{name}.txt").write_text("\n".join(ids) + "\n")
        print(f"{name}: {len(ids)}")

if __name__ == "__main__":
    main()