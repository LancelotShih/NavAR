# AdamW, mixed precision, cosine learning rate, plus a validation check each epoch.
# best checkpoint gets saved

"""Train the U-Net floor segmenter on ADE20K; keep the checkpoint with the best val IoU."""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.dataset import FloorDataset
from src.metrics import FloorMetrics
from src.model import FloorLoss, build_model

REPO = Path(__file__).resolve().parent.parent


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="unet_r34", help="run folder under runs/")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subset", type=int, default=0,
                    help="use only the first N train/val images (quick test)")
    return ap.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    out = REPO / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    train_ds, val_ds = FloorDataset("train"), FloorDataset("val")
    if args.subset:
        train_ds = Subset(train_ds, range(args.subset))
        val_ds = Subset(val_ds, range(min(args.subset, len(val_ds))))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=args.workers,
                              pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=args.workers,
                            pin_memory=True, persistent_workers=True)

    model = build_model().to(device)
    loss_fn = FloorLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_loader))
    scaler = torch.amp.GradScaler()

    def train_one_epoch(model, loader, loss_fn, opt, sched, scaler, device):
        model.train()
        total, n = 0.0, 0
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.float16): # casting to fp16 because I'm training on an RTX 3070
                logits = model(images)
                loss = loss_fn(logits, masks)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

            total += loss.item()
            n += 1
        return total / n

    @torch.no_grad()
    def validate(model, loader, loss_fn, device):
        model.eval()
        metrics = FloorMetrics()
        total = 0.0
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(images)
            total += loss_fn(logits, masks).item()
            metrics.update(logits > 0, masks)
        return {"loss": total / len(loader), **metrics.compute()}

    best_iou = -1.0
    with open(out / "log.csv", "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["epoch", "lr", "train_loss", "val_loss",
                    "val_iou", "val_precision", "val_recall", "seconds"])

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            lr = sched.get_last_lr()[0]
            train_loss = train_one_epoch(model, train_loader, loss_fn, opt, sched, scaler, device)
            val = validate(model, val_loader, loss_fn, device)
            secs = time.time() - t0

            log.writerow([epoch, lr, train_loss, val["loss"],
                        val["iou"], val["precision"], val["recall"], secs])
            f.flush()
            print(f"epoch {epoch:2d}/{args.epochs}  lr {lr:.2e}  train {train_loss:.4f}  "
                f"val {val['loss']:.4f}  IoU {val['iou']:.4f}  "
                f"P {val['precision']:.3f}  R {val['recall']:.3f}  ({secs:.0f}s)", flush=True)

            ckpt = {"epoch": epoch, "model": model.state_dict(), "val": val, "args": vars(args)}
            torch.save(ckpt, out / "last.pt")
            if val["iou"] > best_iou:
                best_iou = val["iou"]
                torch.save(ckpt, out / "best.pt")
                print(f"  new best IoU {best_iou:.4f} → saved best.pt", flush=True)


if __name__ == "__main__":
    main()
