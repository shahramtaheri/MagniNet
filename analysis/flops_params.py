# analysis/flops_params.py
# Compute Params + FLOPs for MagniNet.
#
# Recommended install:
#   pip install fvcore torchinfo
#
# Notes:
# - FLOPs depend on input resolution (default 224x224) and batch size (we use 1).
# - fvcore reports FLOPs as "flop_count_table" (MACs sometimes reported; we keep fvcore output + derived GFLOPs).
# - For Swin/EfficientNet from timm, fvcore usually works, but if it fails you still get params.

import argparse
import os
import torch

from models import MagniNet


def count_params(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": int(total), "trainable_params": int(trainable)}


def try_fvcore_flops(model: torch.nn.Module, x: torch.Tensor):
    try:
        from fvcore.nn import FlopCountAnalysis, flop_count_table
    except Exception as e:
        return None, f"fvcore not available ({e}). Install: pip install fvcore"

    try:
        model.eval()
        with torch.no_grad():
            flops = FlopCountAnalysis(model, x)
            # Total FLOPs (as counted by fvcore)
            total_flops = flops.total()
            table = flop_count_table(flops, max_depth=4)
        return {"total_flops": int(total_flops), "table": table}, None
    except Exception as e:
        return None, f"fvcore flop counting failed: {e}"


def try_torchinfo(model: torch.nn.Module, x: torch.Tensor):
    try:
        from torchinfo import summary
    except Exception as e:
        return None, f"torchinfo not available ({e}). Install: pip install torchinfo"

    try:
        s = summary(
            model,
            input_data=x,
            verbose=0,
            col_names=("input_size", "output_size", "num_params"),
            depth=4,
        )
        return {"torchinfo": str(s)}, None
    except Exception as e:
        return None, f"torchinfo summary failed: {e}"


def parse_args():
    p = argparse.ArgumentParser(description="Compute FLOPs + params for MagniNet.")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--swin_name", type=str, default="swin_tiny_patch4_window7_224")
    p.add_argument("--drop", type=float, default=0.15)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--out_path", type=str, default="./analysis/model_complexity.txt")
    return p.parse_args()


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

    x = torch.randn(1, 3, args.image_size, args.image_size, device=device)

    params = count_params(model)

    flops_res, flops_err = try_fvcore_flops(model, x)
    torchinfo_res, torchinfo_err = try_torchinfo(model, x)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        f.write("MagniNet Complexity Report\n")
        f.write("==========================\n\n")
        f.write(f"Device: {device}\n")
        f.write(f"Input: 1 x 3 x {args.image_size} x {args.image_size}\n")
        f.write(f"dim={args.dim}, heads={args.heads}, swin={args.swin_name}, drop={args.drop}\n\n")

        f.write("Parameters\n")
        f.write("----------\n")
        f.write(f"Total params     : {params['total_params']:,}\n")
        f.write(f"Trainable params : {params['trainable_params']:,}\n\n")

        f.write("FLOPs (fvcore)\n")
        f.write("-------------\n")
        if flops_res is None:
            f.write(f"FLOPs not available: {flops_err}\n\n")
        else:
            total_flops = flops_res["total_flops"]
            gflops = total_flops / 1e9
            f.write(f"Total FLOPs (fvcore count): {total_flops:,}  (~{gflops:.3f} GFLOPs)\n\n")
            f.write(flops_res["table"])
            f.write("\n\n")

        f.write("Model summary (torchinfo)\n")
        f.write("-------------------------\n")
        if torchinfo_res is None:
            f.write(f"torchinfo not available: {torchinfo_err}\n")
        else:
            f.write(torchinfo_res["torchinfo"])
            f.write("\n")

    print(f"Saved: {os.path.abspath(args.out_path)}")
    print(f"Total params: {params['total_params']:,}")
    if flops_res is not None:
        print(f"Total FLOPs: {flops_res['total_flops']:,} (~{flops_res['total_flops']/1e9:.3f} GFLOPs)")
    else:
        print("FLOPs: n/a (see report)")


if __name__ == "__main__":
    main()
