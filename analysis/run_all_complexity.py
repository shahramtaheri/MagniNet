# analysis/run_all_complexity.py
# Generate a combined complexity report (Params, FLOPs, latency/FPS) for MagniNet.
#
# Outputs:
#   analysis/complexity_summary.json
#   analysis/complexity_summary.csv
#   analysis/complexity_summary_latex.txt
#
# Recommended installs:
#   pip install fvcore torchinfo
#
# Usage:
#   python analysis/run_all_complexity.py --device cuda --image_size 224 --batch_size 1 --iters 200 --warmup 50
#
# Notes:
# - FLOPs depend on input size. We use batch=1 for FLOPs.
# - Latency measured as forward pass only with warmup; GPU sync is used for accuracy.

import os
import time
import json
import argparse
import numpy as np
import torch

from models import MagniNet


# -------------------------
# Helpers: Params + FLOPs
# -------------------------
def count_params(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": int(total), "trainable_params": int(trainable)}


def try_fvcore_flops(model: torch.nn.Module, x: torch.Tensor):
    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception as e:
        return None, f"fvcore not available ({e})"

    try:
        model.eval()
        with torch.no_grad():
            flops = FlopCountAnalysis(model, x).total()
        return int(flops), None
    except Exception as e:
        return None, f"fvcore flop counting failed ({e})"


# -------------------------
# Helpers: Inference timing
# -------------------------
@torch.no_grad()
def benchmark_latency(model, x, device: str, iters: int, warmup: int, fp16: bool):
    model.eval()
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

    times_ms = []
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
        times_ms.append((t1 - t0) * 1000.0)

    arr = np.array(times_ms, dtype=np.float64)
    stats = {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "std_ms": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }
    return stats


# -------------------------
# Formatting
# -------------------------
def format_millions(n: int) -> float:
    return n / 1e6


def format_gflops(n: int) -> float:
    return n / 1e9


def to_csv(rows, out_path: str):
    import csv
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def to_latex(rows, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cols = [
        ("Setting", "setting"),
        ("Params (M)", "params_m"),
        ("GFLOPs", "gflops"),
        ("Mean (ms)", "mean_ms"),
        ("P95 (ms)", "p95_ms"),
        ("FPS", "fps"),
    ]
    header = " & ".join([c[0] for c in cols]) + r" \\"
    sep = r"\hline"

    lines = []
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(sep)
    lines.append(header)
    lines.append(sep)
    for r in rows:
        line = " & ".join([
            str(r[c[1]]) for c in cols
        ]) + r" \\"
        lines.append(line)
    lines.append(sep)
    lines.append(r"\end{tabular}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# -------------------------
# Main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Run Params+FLOPs+Latency for MagniNet and export a combined table.")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--fp16", action="store_true", help="Use autocast(fp16) for latency on CUDA.")

    # model
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--swin_name", type=str, default="swin_tiny_patch4_window7_224")
    p.add_argument("--drop", type=float, default=0.15)

    # settings to compare (optional)
    p.add_argument("--settings", type=str, default="default",
                   help="Comma-separated settings. Supported: default,fp16 (fp16 only affects latency).")

    # outputs
    p.add_argument("--out_dir", type=str, default="./analysis")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    if not settings:
        settings = ["default"]

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Build model once (params + FLOPs same across fp32/fp16)
    model = MagniNet(
        num_classes=8,
        dim=args.dim,
        num_heads=args.heads,
        pretrained_backbones=False,
        swin_name=args.swin_name,
        drop=args.drop,
    ).to(device)

    # Params
    pcount = count_params(model)

    # FLOPs (batch=1)
    x_flops = torch.randn(1, 3, args.image_size, args.image_size, device=device)
    flops, flops_err = try_fvcore_flops(model, x_flops)

    # Latency per setting
    rows = []
    for s in settings:
        fp16 = (s.lower() == "fp16") or (args.fp16 and s.lower() == "default_fp16")
        fp16 = bool(fp16 and device == "cuda")

        x = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
        stats = benchmark_latency(
            model=model,
            x=x,
            device=device,
            iters=args.iters,
            warmup=args.warmup,
            fp16=fp16,
        )
        mean_s = stats["mean_ms"] / 1000.0
        fps = (args.batch_size / mean_s) if mean_s > 0 else 0.0

        row = {
            "setting": s,
            "device": device,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "params_total": pcount["total_params"],
            "params_trainable": pcount["trainable_params"],
            "params_m": round(format_millions(pcount["total_params"]), 3),
            "flops_total": flops if flops is not None else None,
            "gflops": round(format_gflops(flops), 3) if flops is not None else None,
            "mean_ms": round(stats["mean_ms"], 3),
            "median_ms": round(stats["median_ms"], 3),
            "p95_ms": round(stats["p95_ms"], 3),
            "std_ms": round(stats["std_ms"], 3),
            "fps": round(fps, 2),
            "fp16": fp16,
            "flops_note": None if flops is not None else flops_err,
        }
        rows.append(row)

    # Save JSON/CSV/LaTeX
    json_path = os.path.join(out_dir, "complexity_summary.json")
    csv_path = os.path.join(out_dir, "complexity_summary.csv")
    tex_path = os.path.join(out_dir, "complexity_summary_latex.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": {
                    "dim": args.dim,
                    "heads": args.heads,
                    "swin_name": args.swin_name,
                    "drop": args.drop,
                },
                "params": pcount,
                "flops_total": flops,
                "rows": rows,
            },
            f,
            indent=2,
        )

    to_csv(rows, csv_path)
    # For LaTeX, keep only the most relevant columns and round nicely
    latex_rows = []
    for r in rows:
        latex_rows.append({
            "setting": r["setting"],
            "params_m": f'{r["params_m"]:.3f}',
            "gflops": f'{r["gflops"]:.3f}' if r["gflops"] is not None else "n/a",
            "mean_ms": f'{r["mean_ms"]:.3f}',
            "p95_ms": f'{r["p95_ms"]:.3f}',
            "fps": f'{r["fps"]:.2f}',
        })
    to_latex(latex_rows, tex_path)

    # Console summary
    print("\n=== MagniNet Complexity Summary ===")
    print(f"Device: {device}")
    print(f"Input: {args.batch_size} x 3 x {args.image_size} x {args.image_size}")
    print(f"Params: {pcount['total_params']:,} ({format_millions(pcount['total_params']):.3f} M)")
    if flops is not None:
        print(f"FLOPs: {flops:,} (~{format_gflops(flops):.3f} GFLOPs)")
    else:
        print(f"FLOPs: n/a ({flops_err})")
    for r in rows:
        print(f"- {r['setting']}: mean={r['mean_ms']:.3f} ms | p95={r['p95_ms']:.3f} ms | FPS={r['fps']:.2f} | fp16={r['fp16']}")

    print("\nSaved:")
    print(f"- {os.path.abspath(json_path)}")
    print(f"- {os.path.abspath(csv_path)}")
    print(f"- {os.path.abspath(tex_path)}")


if __name__ == "__main__":
    main()
