"""Sanity-check the PyTorch + CUDA setup and run a small fp16 matmul under autocast."""

import sys

import torch


def main() -> int:
    print(f"torch version:  {torch.__version__} (CUDA build {torch.version.cuda})")
    available = torch.cuda.is_available()
    print(f"CUDA available: {available}")
    if not available:
        return 1

    props = torch.cuda.get_device_properties(0)
    print(f"GPU:            {props.name}")
    print(f"GPU memory:     {props.total_memory / 1024**3:.1f} GiB")
    print(f"Compute cap.:   {props.major}.{props.minor}")

    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        c = a @ b
    torch.cuda.synchronize()

    ref = a.double() @ b.double()
    rel_err = ((c.double() - ref).norm() / ref.norm()).item()
    print(f"autocast matmul: dtype={c.dtype}, shape={tuple(c.shape)}, rel. error vs fp64={rel_err:.2e}")

    ok = c.dtype == torch.float16 and rel_err < 1e-2
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
