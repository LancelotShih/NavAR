"""Run the floor model on a video, save an overlay video, and time every step to check 30 FPS.

Usage (from the repo root):
    python -m src.video walk.mp4                      # writes outputs/video/walk_floor.mp4
    python -m src.video walk.mp4 --no-write           # timing only, no output video

Each frame goes through the same steps as src/predict.py: resize so the short side is 256,
pad to a multiple of 32, run the U-Net, then scale the floor probability back to full size.
Times are printed per step (decode, model, overlay, encode) so you can tell how fast the model
path is versus the parts that only exist for this demo (drawing and saving the video).
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import imageio_ffmpeg  # bundles an ffmpeg binary, so no system install is needed
import numpy as np
import torch

from src.dataset import build_transform
from src.model import build_model
from src.predict import best_checkpoint, predict

REPO = Path(__file__).resolve().parent.parent
STEPS = ("decode", "model", "overlay", "encode")


def overlay(frame_bgr: np.ndarray, floor: np.ndarray) -> np.ndarray:
    # half-transparent green over predicted floor (same look as src/predict.py's tint).
    # Blend the whole frame, then copy only the floor pixels across: much faster than
    # numpy boolean indexing on a 1080p frame.
    blended = cv2.addWeighted(frame_bgr, 0.5, np.full_like(frame_bgr, (0, 255, 0)), 0.5, 0)
    out = frame_bgr.copy()
    cv2.copyTo(blended, floor.astype(np.uint8), out)
    return out


class H264Writer:
    # OpenCV's pip build can only save old MPEG-4 ("mp4v"), which VS Code, browsers and many
    # Windows players can't open, so pipe the frames to ffmpeg and save standard H.264 instead
    def __init__(self, path: Path, w: int, h: int, fps: float):
        self.proc = subprocess.Popen(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
             "-pix_fmt", "yuv420p",        # the pixel format every player supports
             "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # yuv420p needs even width and height
             "-movflags", "+faststart",    # index at the front so players can start right away
             str(path)],
            stdin=subprocess.PIPE)
        assert self.proc.stdin is not None  # always a pipe with stdin=PIPE; tells the type checker
        self.stdin = self.proc.stdin

    def write(self, frame_bgr: np.ndarray) -> None:
        self.stdin.write(np.ascontiguousarray(frame_bgr).tobytes())

    def release(self) -> None:
        self.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed to write the video")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--ckpt", type=Path, help="checkpoint (default: best val IoU under runs/)")
    ap.add_argument("--out", type=Path, help="default: outputs/video/<name>_floor.mp4")
    ap.add_argument("--threshold", type=float, default=0.5, help="floor if probability > this")
    ap.add_argument("--no-write", action="store_true", help="time only; don't draw or save a video")
    ap.add_argument("--warmup", type=int, default=10, help="frames left out of the timing")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.ckpt or best_checkpoint()
    model = build_model(pretrained=False).to(device).eval()
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model"])
    transform = build_transform(train=False)

    cap = cv2.VideoCapture(str(args.video))  # applies the phone's rotation tag by default
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    h = w = 0  # set from the first frame; an empty video exits before they're used
    out_path = args.out or REPO / "outputs" / "video" / f"{args.video.stem}_floor.mp4"
    times = {s: [] for s in STEPS}
    floor_share = []
    i = 0
    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        t1 = time.perf_counter()
        # predict() ends with .cpu(), which waits for the GPU, so this time includes the GPU work
        floor = predict(model, transform, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), device) > args.threshold
        t2 = time.perf_counter()
        t3 = t4 = t2
        if not args.no_write:
            shown = overlay(frame, floor)
            t3 = time.perf_counter()
            if writer is None:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                writer = H264Writer(out_path, w, h, src_fps)
            writer.write(shown)
            t4 = time.perf_counter()
        if i >= args.warmup:  # the first frames include CUDA start-up
            for s, dt in zip(STEPS, (t1 - t0, t2 - t1, t3 - t2, t4 - t3)):
                times[s].append(dt * 1000)
        floor_share.append(floor.mean())
        i += 1
    cap.release()
    if writer is not None:
        writer.release()
    if not times["model"]:
        raise SystemExit(f"only {i} frames read; need more than --warmup {args.warmup}")

    ms = {s: float(np.mean(v)) for s, v in times.items()}
    model_p95 = float(np.percentile(times["model"], 95))
    total = sum(ms.values())
    print(f"{args.video.name}: {i} frames ({n_frames} reported), {w}x{h} at {src_fps:.1f} FPS, on {device}")
    print(f"model: {ckpt_path}")
    print(f"\n{'step':<10}{'ms/frame':>10}")
    for s in STEPS:
        if s in ("overlay", "encode") and args.no_write:
            continue
        print(f"{s:<10}{ms[s]:>10.1f}")
    print(f"\nmodel path (resize + U-Net + upscale): {1000 / ms['model']:.0f} FPS "
          f"(95% of frames under {model_p95:.1f} ms; 30 FPS needs 33.3 ms)")
    print(f"end to end incl. decode{'' if args.no_write else ', overlay, encode'}: {1000 / total:.0f} FPS")

    summary = {"video": str(args.video), "frames": i, "size": [w, h], "source_fps": src_fps,
               "device": str(device), "checkpoint": str(ckpt_path), "ms_per_frame": ms,
               "model_p95_ms": model_p95, "model_fps": 1000 / ms["model"], "end_to_end_fps": 1000 / total,
               "mean_floor_share": float(np.mean(floor_share)), "wrote_video": not args.no_write}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    if writer is not None:
        print(f"\nwrote {out_path} and {out_path.with_suffix('.json').name}")


if __name__ == "__main__":
    main()
