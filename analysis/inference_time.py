# analysis/inference_time.py
# Measure inference latency for MagniNet (forward pass only).
#
# Key points for a credible report:
# - warmup iterations (to stabilize kernels/caches)
# - torch.cuda.synchronize() around timing (for GPU accuracy)
# - measure multiple runs and report mean/median/p95 + FPS
#
# Usage:
#   python analysis/inference_time.py --device cuda --image_size 224 --iters 200 --warmup 50

import argparse
import os
import time
import numpy as np
import torch

from models import MagniNet


def parse_args():
    p = argparse.ArgumentParser(description="Inference time benchmark for MagniNet.")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--swin_name", type=str, default="swin_tiny_patch4_window7_224")
    p.add_argument("--drop", type=float, default=0.15)
    p.add_argument("--fp16", action="store_true", help="Use autocast(fp16) on CUDA.")
    p.add_argument("--out_path", type=str, default="./analysis/inference_time.txt")
    return p.parse_args()


@torch.no_grad()
def benchmark(model, x, device: str, iters: int, warmup: int, fp16: bool):
    model.eval()
    times = []

    use_cuda = (device == "cuda") and torch.cuda.is_available()

    # Warmup
    for _ in range(warmup):
        if use_cuda and fp16:
            with torch.cuda.amp.autocast():
                _ = model(x)
        else:
            _ = model(x)

    if use_cuda:
        torch.cuda.synchronize()

    # Timed runs
    for _ in range(iters):
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        if use_cuda and fp16:
            with torch.cuda.amp.autocast():
                _ = model(x)
        else:
            _ = model(x)

        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms

    times = np.array(times, dtype=np.float64)
    stats = {
        "mean_ms": float(times.mean()),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "std_ms": float(times.std(ddof=1)) if len(times) > 1 else 0.0,
        "min_ms": float(times.min()),
        "max_ms": float(times.max()),
    }
    return stats, times


def main():
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = MagniNet(
        num_classes=8,
        dim=args.dim,
        num_heads=args.heads,
        pretrained_backbones=False,
        swin_name=args.swin_name,
        drop=args.drop,
    ).to(device)

    x = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)

    stats, _ = benchmark(
        model=model,
        x=x,
        device=device,
        iters=args.iters,
        warmup=args.warmup,
        fp16=args.fp16,
    )

    # FPS computed from mean latency
    mean_s = stats["mean_ms"] / 1000.0
    fps = (args.batch_size / mean_s) if mean_s > 0 else 0.0

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        f.write("MagniNet Inference Time Report\n")
        f.write("===============================\n\n")
        f.write(f"Device     : {device}\n")
        f.write(f"Input      : {args.batch_size} x 3 x {args.image_size} x {args.image_size}\n")
        f.write(f"FP16       : {bool(args.fp16 and device == 'cuda')}\n")
        f.write(f"iters      : {args.iters}\n")
        f.write(f"warmup     : {args.warmup}\n")
        f.write(f"dim={args.dim}, heads={args.heads}, swin={args.swin_name}, drop={args.drop}\n\n")

        f.write("Latency (ms)\n")
        f.write("------------\n")
        f.write(f"Mean   : {stats['mean_ms']:.3f}\n")
        f.write(f"Median : {stats['median_ms']:.3f}\n")
        f.write(f"P95    : {stats['p95_ms']:.3f}\n")
        f.write(f"Std    : {stats['std_ms']:.3f}\n")
        f.write(f"Min    : {stats['min_ms']:.3f}\n")
        f.write(f"Max    : {stats['max_ms']:.3f}\n\n")

        f.write("Throughput\n")
        f.write("----------\n")
        f.write(f"FPS (from mean) : {fps:.2f}\n")

    print(f"Saved: {os.path.abspath(args.out_path)}")
    print(f"Mean latency: {stats['mean_ms']:.3f} ms | P95: {stats['p95_ms']:.3f} ms | FPS: {fps:.2f}")


if __name__ == "__main__":
    main()
