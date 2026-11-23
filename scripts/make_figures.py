"""Regenerate every figure and number table used by the progress report.

Reads only saved results (masks, runs/*/log.csv, runs/sweep_summary.csv,
runs/baseline/*, outputs/eval_<split>/*) plus one forward pass per example image,
and writes into documentation/progress_report/{figures,tables}/.

Usage (from the repo root, after src.baseline and src.evaluate have been run):
    python scripts/make_figures.py
"""

import argparse
import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.baseline import OUT as BASELINE_DIR, baseline_transform, to_rgb  # noqa: E402
from src.dataset import IGNORE, IMAGES, MASKS, FOLDER, FloorDataset       # noqa: E402
from src.evaluate import image_iou, predict_both                          # noqa: E402
from src.model import build_model                                         # noqa: E402
from src.predict import tint                                              # noqa: E402

DOC = REPO / "documentation" / "progress_report"
FIG, TAB = DOC / "figures", DOC / "tables"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})
RUN_LABELS = {  # sweep run -> label used in the report
    "r34_lr1e-4_bs16": r"lr $10^{-4}$, batch 16",
    "r34_lr3e-4_bs16": r"lr $3{\times}10^{-4}$, batch 16",
    "r34_lr3e-4_bs32": r"lr $3{\times}10^{-4}$, batch 32",
    "r34_lr1e-3_bs16": r"lr $10^{-3}$, batch 16",
}
FLOOR_RGB, NOT_RGB, IGNORE_RGB = (0, 200, 0), (60, 60, 60), (220, 40, 40)


# ---------------------------------------------------------------- data statistics

def _mask_counts(path: Path) -> tuple[int, int, int]:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return int((m == 0).sum()), int((m == 1).sum()), int((m == IGNORE).sum())


def data_table() -> None:
    rows = []
    for split, title in (("train", "Train"), ("val", "Validation"), ("test", "Test (held out)")):
        ids = (REPO / "splits" / f"{split}.txt").read_text().split()
        paths = [MASKS / FOLDER[split] / f"{i}.png" for i in ids]
        with Pool() as pool:
            counts = np.array(pool.map(_mask_counts, paths, chunksize=64), dtype=np.int64)
        total = counts.sum()
        no_floor = (counts[:, 1] == 0).mean()
        rows.append((title, len(ids), no_floor, *(counts.sum(0) / total)))
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Split & Images & No floor & Floor px & Not-floor px & Ignored px \\", r"\midrule"]
    for title, n, nf, nfp, fp, ig in rows:
        lines.append(f"{title} & {n:,} & {nf:.1%} & {fp:.1%} & {nfp:.1%} & {ig:.1%} \\\\".replace("%", r"\%"))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "data.tex").write_text("\n".join(lines) + "\n")


def data_samples(ids: list[str]) -> None:
    """Photo and its cleaned 3-value mask, for a few training images."""
    fig, axes = plt.subplots(1, 2 * len(ids), figsize=(6.9, 1.12))
    for k, id_ in enumerate(ids):
        rgb = cv2.cvtColor(cv2.imread(str(IMAGES / "training" / f"{id_}.jpg")), cv2.COLOR_BGR2RGB)
        m = cv2.imread(str(MASKS / "training" / f"{id_}.png"), cv2.IMREAD_UNCHANGED)
        colour = np.zeros_like(rgb)
        colour[m == 0], colour[m == 1], colour[m == IGNORE] = NOT_RGB, FLOOR_RGB, IGNORE_RGB
        for ax, img in zip(axes[2 * k: 2 * k + 2], (rgb, colour)):
            ax.imshow(img)
            ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=np.array(c) / 255) for c in (FLOOR_RGB, NOT_RGB, IGNORE_RGB)]
    fig.legend(handles, ["floor", "not floor", "ignore"], loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout(pad=0.2, w_pad=0.3)
    fig.savefig(FIG / "data_samples.pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- training results

def read_log(name: str) -> list[dict]:
    with open(REPO / "runs" / name / "log.csv") as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def curves(best: str) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 1.6))
    log = read_log(best)
    ep = [r["epoch"] for r in log]
    a.plot(ep, [r["train_loss"] for r in log], label="train")
    a.plot(ep, [r["val_loss"] for r in log], label="validation")
    a.set(xlabel="epoch", ylabel="BCE + Dice loss", title=f"(a) loss, best run ({RUN_LABELS[best]})")
    a.legend(frameon=False)
    for name, label in RUN_LABELS.items():
        log = read_log(name)
        best_iou = max(r["val_iou"] for r in log)
        b.plot([r["epoch"] for r in log], [r["val_iou"] for r in log], label=f"{label}  ({best_iou:.3f})",
               lw=1.6 if name == best else 1.0)
    b.set(xlabel="epoch", ylabel="validation floor IoU", title="(b) validation IoU, sweep (best IoU)")
    b.set_ylim(0.3, 0.76)
    b.legend(frameon=False, loc="lower right", ncol=1)
    fig.tight_layout(pad=0.3, w_pad=1.5)
    fig.savefig(FIG / "curves.pdf", bbox_inches="tight")
    plt.close(fig)


def sweep_table() -> None:
    with open(REPO / "runs" / "sweep_summary.csv") as f:
        rows = list(csv.DictReader(f))
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Run & Val IoU & Precision & Recall & Best epoch \\", r"\midrule"]
    for i, r in enumerate(rows):
        cells = [RUN_LABELS[r["name"]], f"{float(r['val_iou']):.3f}", f"{float(r['val_precision']):.3f}",
                 f"{float(r['val_recall']):.3f}", r["best_epoch"]]
        if i == 0:
            cells = [rf"\textbf{{{c}}}" for c in cells]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "sweep.tex").write_text("\n".join(lines) + "\n")


def threshold_curve() -> None:
    with open(BASELINE_DIR / "threshold_sweep_val.csv") as f:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    t = [r["threshold"] for r in rows]
    fig, ax = plt.subplots(figsize=(2.6, 1.3))
    for key, label in (("iou", "IoU"), ("precision", "precision"), ("recall", "recall")):
        ax.plot(t, [r[key] for r in rows], marker="o", ms=2, label=label)
    best = max(rows, key=lambda r: r["iou"])
    ax.axvline(best["threshold"], color="0.6", lw=0.8, ls="--")
    ax.set(xscale="log", xlabel=r"threshold $\tau$ on MSE", ylim=(0, 1))
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(pad=0.2)
    fig.savefig(FIG / "threshold.pdf", bbox_inches="tight")
    plt.close(fig)


def results_table(split: str) -> None:
    r = json.loads((REPO / "outputs" / f"eval_{split}" / "results.json").read_text())
    pct = lambda v: f"{100 * v:.1f}"
    lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r"& & & & \multicolumn{4}{c}{Recall by surface (\%)} & No-floor imgs \\ \cmidrule(lr){5-8}",
             r"Method & IoU & Prec. & Rec. & floor & rug & sidewalk & path & w/ false floor \\", r"\midrule"]
    for key, name in (("baseline", "Colour baseline"), ("unet", "U-Net (ours)")):
        m = r[key]
        rc = m["recall_by_class"]
        cells = [name, f"{m['iou']:.3f}", f"{m['precision']:.3f}", f"{m['recall']:.3f}",
                 pct(rc["floor"]), pct(rc["rug"]), pct(rc["sidewalk"]), pct(rc["path"]),
                 pct(m["no_floor_images_with_false_floor"]) + r"\%"]
        if key == "unet":
            cells = [rf"\textbf{{{c}}}" for c in cells]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "results.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- qualitative figures

def load_models(ckpt_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(pretrained=False).to(device).eval()
    model.load_state_dict(ckpt["model"])
    threshold = json.loads((BASELINE_DIR / "threshold.json").read_text())["threshold"]
    return model, threshold, device


def run_example(split, id_, model, threshold, device):
    model_ds, base_ds = FloorDataset(split), FloorDataset(split, transform=baseline_transform())
    i = model_ds.ids.index(id_)
    x, _ = model_ds[i]
    image, mask = base_ds[i]
    h, w = mask.shape
    preds = predict_both(model, threshold, x, image, h, w, device)
    return to_rgb(image), mask, preds


def qualitative(split, ids, model, threshold, device) -> None:
    rows = ("label", "baseline", "unet")
    titles = {"label": "ground truth", "baseline": "colour baseline", "unet": "U-Net"}
    fig, axes = plt.subplots(len(rows), len(ids), figsize=(6.9, 2.75), squeeze=False)
    for col, id_ in enumerate(ids):
        rgb, mask, preds = run_example(split, id_, model, threshold, device)
        m = mask.numpy()
        for row, key in enumerate(rows):
            ax = axes[row, col]
            if key == "label":
                img = tint(tint(rgb, m == 1, FLOOR_RGB), m == IGNORE, IGNORE_RGB)
                text = ""
            else:
                img = tint(rgb, preds[key].numpy(), FLOOR_RGB)
                iou = image_iou(preds[key], mask)
                text = f"IoU {iou:.2f}" if iou is not None else ""
            ax.imshow(img, aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if text:
                ax.text(0.03, 0.05, text, transform=ax.transAxes, fontsize=7, color="white",
                        bbox=dict(facecolor="black", alpha=0.6, pad=1.2, lw=0))
            if col == 0:
                ax.set_ylabel(titles[key], fontsize=8)
    fig.tight_layout(pad=0.2, h_pad=0.25, w_pad=0.25)
    fig.savefig(FIG / "qualitative.pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)


def overview(split, id_, model, threshold, device) -> None:
    """Input photo and U-Net output for the Figure 1 diagram (PNG, placed by TikZ)."""
    rgb, _, preds = run_example(split, id_, model, threshold, device)
    cv2.imwrite(str(FIG / "overview_input.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    out = tint(rgb, preds["unet"].numpy(), FLOOR_RGB)
    cv2.imwrite(str(FIG / "overview_output.jpg"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])


def pick_examples(split: str) -> list[str]:
    """Val images at the 90th / 50th / 10th percentile of U-Net IoU (as in src.evaluate)."""
    with open(REPO / "outputs" / f"eval_{split}" / "per_image.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["unet"] and float(r["floor_frac"]) > 0.05]
    rows.sort(key=lambda r: float(r["unet"]))
    return [rows[int(q * (len(rows) - 1))]["id"] for q in (0.9, 0.5, 0.1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--best", default="r34_lr1e-4_bs16", help="run shown as the primary model")
    ap.add_argument("--extra", nargs="*", default=["ADE_train_00016867", "ADE_train_00017297"],
                    help="extra example ids for the qualitative figure (must be in --split)")
    ap.add_argument("--samples", nargs="*", default=["ADE_train_00000654", "ADE_train_00015036"],
                    help="training ids shown as cleaned data samples")
    ap.add_argument("--overview", default="ADE_train_00016867", help="id used in Figure 1")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)

    print("tables ...")
    data_table()
    sweep_table()
    results_table(args.split)
    print("plots ...")
    data_samples(args.samples)
    curves(args.best)
    threshold_curve()
    print("qualitative examples ...")
    model, threshold, device = load_models(REPO / "runs" / args.best / "best.pt")
    qualitative(args.split, pick_examples(args.split) + args.extra, model, threshold, device)
    overview(args.split, args.overview, model, threshold, device)
    print(f"wrote {FIG.relative_to(REPO)}/ and {TAB.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
