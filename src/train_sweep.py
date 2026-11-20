"""Run several training configs one after another, then rank them by best val IoU.

Usage (from the repo root):
    nohup .venv/bin/python -m src.train_sweep > runs/sweep.out 2>&1 &
    tail -f runs/sweep.out

Configs that already finished are skipped, so re-running the same command after a
crash or after adding new configs only trains what is missing.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"

# run name -> extra flags for src.train (anything not listed uses train.py's default)
SWEEP = {
    "r34_lr3e-4_bs16": [],                          # the defaults
    "r34_lr1e-4_bs16": ["--lr", "1e-4"],
    "r34_lr1e-3_bs16": ["--lr", "1e-3"],
    "r34_lr3e-4_bs32": ["--batch-size", "32"],
}


def finished(name: str) -> bool:
    """True if the run's log has one row for every epoch it asked for."""
    args_file, log_file = RUNS / name / "args.json", RUNS / name / "log.csv"
    if not (args_file.exists() and log_file.exists()):
        return False
    epochs = json.loads(args_file.read_text())["epochs"]
    with open(log_file) as f:
        return sum(1 for _ in f) - 1 >= epochs      # minus 1 for the header row


def run(name: str, extra: list[str]) -> int:
    """Train one config in its own process; its output goes to runs/<name>/train.out."""
    (RUNS / name).mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "src.train", "--name", name, *extra]
    with open(RUNS / name / "train.out", "w") as out:
        return subprocess.run(cmd, cwd=REPO, stdout=out, stderr=subprocess.STDOUT).returncode


def summarize(names: list[str]) -> None:
    """Rank runs by best val IoU; print a table and write runs/sweep_summary.csv."""
    rows = []
    for name in names:
        log_file = RUNS / name / "log.csv"
        if not log_file.exists():
            continue
        with open(log_file) as f:
            epochs = list(csv.DictReader(f))
        if not epochs:
            continue
        best = max(epochs, key=lambda r: float(r["val_iou"]))
        rows.append({
            "name": name,
            "val_iou": float(best["val_iou"]),
            "val_precision": float(best["val_precision"]),
            "val_recall": float(best["val_recall"]),
            "best_epoch": int(best["epoch"]),
            "epochs_run": len(epochs),
            "minutes": sum(float(r["seconds"]) for r in epochs) / 60,
        })
    rows.sort(key=lambda r: r["val_iou"], reverse=True)

    if not rows:
        print("\nno finished epochs to summarize yet")
        return

    with open(RUNS / "sweep_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'run':<22}{'IoU':>8}{'P':>8}{'R':>8}{'best ep':>9}{'epochs':>8}{'min':>6}")
    for r in rows:
        print(f"{r['name']:<22}{r['val_iou']:>8.4f}{r['val_precision']:>8.3f}{r['val_recall']:>8.3f}"
              f"{r['best_epoch']:>9}{r['epochs_run']:>8}{r['minutes']:>6.0f}")
    print(f"\nwrote {RUNS / 'sweep_summary.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="run only these config names")
    ap.add_argument("--rerun", action="store_true", help="also re-run finished configs")
    args = ap.parse_args()

    names = args.only or list(SWEEP)
    unknown = [n for n in names if n not in SWEEP]
    if unknown:
        sys.exit(f"unknown config(s): {unknown}; choose from {list(SWEEP)}")

    for i, name in enumerate(names, 1):
        tag = f"[{i}/{len(names)}] {name}"
        if finished(name) and not args.rerun:
            print(f"{tag}: already finished, skipping", flush=True)
            continue
        print(f"{tag}: training...", flush=True)
        t0 = time.time()
        code = run(name, SWEEP[name])
        status = "done" if code == 0 else f"FAILED (exit {code}), see runs/{name}/train.out"
        print(f"{tag}: {status} after {(time.time() - t0) / 60:.0f} min", flush=True)

    summarize(list(SWEEP))


if __name__ == "__main__":
    main()
