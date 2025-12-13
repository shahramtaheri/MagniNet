# infer.py
# Inference script for MagniNet:
# - Single image or folder inference
# - Outputs: predicted 8-class label + probability, predicted binary label + probability
# - Saves a JSONL file with per-image predictions
#
# Usage examples:
#   python infer.py --checkpoint ./runs/magninet_40x/best.pt --input /path/img.jpg
#   python infer.py --checkpoint ./runs/magninet_40x/best.pt --input /path/images_dir --out_dir ./results/infer
#
# pip install timm pillow

import os
import json
import argparse
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError as e:
    raise ImportError("Please install timm: pip install timm") from e


# -------------------------
# MagniNet model (same as train/test scripts)
# -------------------------
class AdaptiveGate(nn.Module):
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        g = self.mlp(x)
        return x * g


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, attn_drop: float = 0.0, proj_drop: float = 0.15):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.gate = AdaptiveGate(dim)

    def forward(self, global_tokens, local_tokens):
        B, Ng, D = global_tokens.shape
        _, Nl, _ = local_tokens.shape

        q = self.q(global_tokens).reshape(B, Ng, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(local_tokens).reshape(B, Nl, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(local_tokens).reshape(B, Nl, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, Ng, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        out = self.gate(out)
        return out


class FFN(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.15):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.net(x)


class AdaptiveAttentionTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, attn_drop: float = 0.0, drop: float = 0.15):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True)
        self.drop1 = nn.Dropout(drop)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FFN(dim, drop=drop)
        self.gate = AdaptiveGate(dim)

    def forward(self, x):
        x1 = self.norm1(x)
        attn_out, _ = self.attn(x1, x1, x1, need_weights=False)
        x = x + self.drop1(attn_out)
        x = x + self.ffn(self.norm2(x))
        x = self.gate(x)
        return x


class EfficientNetLocalEncoder(nn.Module):
    def __init__(self, model_name: str = "efficientnet_b0", out_dim: int = 256, pretrained: bool = False):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(2, 4, 6),
        )
        feat_channels = self.backbone.feature_info.channels()
        self.proj = nn.ModuleList([nn.Conv2d(c, out_dim, kernel_size=1, bias=False) for c in feat_channels])
        self.bn = nn.ModuleList([nn.BatchNorm2d(out_dim) for _ in feat_channels])
        self.fuse = nn.Conv2d(out_dim * len(feat_channels), out_dim, kernel_size=1, bias=False)

    def forward(self, x):
        feats = self.backbone(x)
        proj_feats = []
        for f, p, bn in zip(feats, self.proj, self.bn):
            proj_feats.append(bn(p(f)))

        Hmax = max(t.shape[-2] for t in proj_feats)
        Wmax = max(t.shape[-1] for t in proj_feats)
        aligned = []
        for t in proj_feats:
            if t.shape[-2:] != (Hmax, Wmax):
                t = F.interpolate(t, size=(Hmax, Wmax), mode="bilinear", align_corners=False)
            aligned.append(t)

        fused = self.fuse(torch.cat(aligned, dim=1))
        tokens = fused.flatten(2).transpose(1, 2)
        return tokens


class MagniNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        dim: int = 256,
        num_heads: int = 8,
        pretrained_backbones: bool = False,
        swin_name: str = "swin_tiny_patch4_window7_224",
        drop: float = 0.15,
    ):
        super().__init__()
        self.local_encoder = EfficientNetLocalEncoder("efficientnet_b0", out_dim=dim, pretrained=pretrained_backbones)

        self.swin = timm.create_model(
            swin_name,
            pretrained=pretrained_backbones,
            features_only=True,
            out_indices=(3,),
        )
        swin_channels = self.swin.feature_info.channels()[0]
        self.swin_proj = nn.Sequential(
            nn.Conv2d(swin_channels, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )

        self.cross_attn = CrossAttentionFusion(dim=dim, num_heads=num_heads, attn_drop=0.0, proj_drop=drop)
        self.blocks = nn.Sequential(
            AdaptiveAttentionTransformerBlock(dim=dim, num_heads=num_heads, attn_drop=0.0, drop=drop),
            AdaptiveAttentionTransformerBlock(dim=dim, num_heads=num_heads, attn_drop=0.0, drop=drop),
        )
        self.norm = nn.LayerNorm(dim)

        self.head_multiclass = nn.Linear(dim, num_classes)
        self.head_binary = nn.Linear(dim, 1)

    def forward(self, x):
        local_tokens = self.local_encoder(x)
        swin_feat = self.swin(x)[0]
        swin_feat = self.swin_proj(swin_feat)
        global_tokens = swin_feat.flatten(2).transpose(1, 2)

        fused = self.cross_attn(global_tokens, local_tokens)
        fused = self.blocks(fused)
        fused = self.norm(fused)

        emb = fused.mean(dim=1)
        logits_mc = self.head_multiclass(emb)
        logits_bin = self.head_binary(emb).squeeze(-1)
        return logits_mc, logits_bin


# -------------------------
# Utils
# -------------------------
def list_images(path: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
    files = []
    for root, _, fnames in os.walk(path):
        for f in fnames:
            if f.lower().endswith(exts):
                files.append(os.path.join(root, f))
    return sorted(files)


def load_checkpoint(model: nn.Module, ckpt_path: str, device: str):
    obj = torch.load(ckpt_path, map_location=device)
    if isinstance(obj, dict) and "model" in obj:
        state = obj["model"]
    else:
        state = obj
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)}")


def preprocess_image(img_path: str, image_size: int) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    img = img.resize((image_size, image_size), resample=Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()  # [1,3,H,W]
    return x


def softmax_probs(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits, dim=1).detach().cpu().numpy()


def sigmoid_probs(logits: torch.Tensor) -> np.ndarray:
    return torch.sigmoid(logits).detach().cpu().numpy()


# -------------------------
# Inference
# -------------------------
@torch.no_grad()
def run_inference(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    # class names (edit if your indexing differs)
    default_names = ["AD", "FA", "PT", "TA", "DC", "LC", "MC", "PC"]
    class_names = args.class_names.split(",") if args.class_names else default_names
    if len(class_names) != args.num_classes:
        print(f"[WARN] class_names count ({len(class_names)}) != num_classes ({args.num_classes}); using indices.")
        class_names = [str(i) for i in range(args.num_classes)]

    model = MagniNet(
        num_classes=args.num_classes,
        dim=args.dim,
        num_heads=args.heads,
        pretrained_backbones=False,
        swin_name=args.swin_name,
        drop=args.drop,
    ).to(device)
    model.eval()
    load_checkpoint(model, args.checkpoint, device)

    # input can be a file or folder
    if os.path.isdir(args.input):
        img_paths = list_images(args.input)
        if len(img_paths) == 0:
            raise RuntimeError(f"No images found under: {args.input}")
    else:
        img_paths = [args.input]

    out_jsonl = os.path.join(args.out_dir, "predictions.jsonl")
    results = []

    for img_path in img_paths:
        x = preprocess_image(img_path, args.image_size).to(device)

        logits_mc, logits_bin = model(x)

        probs_mc = softmax_probs(logits_mc)[0]   # [8]
        probs_bin = float(sigmoid_probs(logits_bin)[0])  # scalar

        pred_mc_idx = int(np.argmax(probs_mc))
        pred_mc_name = class_names[pred_mc_idx]
        pred_mc_prob = float(probs_mc[pred_mc_idx])

        pred_bin = int(probs_bin >= 0.5)
        pred_bin_name = "Malignant" if pred_bin == 1 else "Benign"

        rec = {
            "image_path": os.path.abspath(img_path),
            "multiclass_pred_index": pred_mc_idx,
            "multiclass_pred_name": pred_mc_name,
            "multiclass_pred_prob": pred_mc_prob,
            "multiclass_probs": {class_names[i]: float(probs_mc[i]) for i in range(args.num_classes)},
            "binary_pred": pred_bin,
            "binary_pred_name": pred_bin_name,
            "binary_prob_malignant": probs_bin,
            "checkpoint": os.path.abspath(args.checkpoint),
        }
        results.append(rec)

    # write JSONL
    with open(out_jsonl, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # print a compact summary (first few)
    print(f"\nDevice: {device}")
    print(f"Checkpoint: {os.path.abspath(args.checkpoint)}")
    print(f"Processed {len(results)} image(s). Saved: {out_jsonl}\n")

    k = min(len(results), 10)
    for i in range(k):
        r = results[i]
        print(
            f"[{i+1:02d}] {os.path.basename(r['image_path'])} | "
            f"8-class: {r['multiclass_pred_name']} ({r['multiclass_pred_prob']:.3f}) | "
            f"binary: {r['binary_pred_name']} (P(mal)={r['binary_prob_malignant']:.3f})"
        )
    if len(results) > k:
        print(f"... ({len(results)-k} more)")


def parse_args():
    p = argparse.ArgumentParser(description="MagniNet inference on a single image or a folder.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (best.pt/last.pt/state_dict).")
    p.add_argument("--input", type=str, required=True, help="Path to an image file or a folder of images.")
    p.add_argument("--out_dir", type=str, default="./results/infer", help="Output directory.")
    p.add_argument("--cpu", action="store_true", help="Force CPU inference.")

    # model params (must match training)
    p.add_argument("--num_classes", type=int, default=8)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--swin_name", type=str, default="swin_tiny_patch4_window7_224")
    p.add_argument("--drop", type=float, default=0.15)

    # preprocessing
    p.add_argument("--image_size", type=int, default=224)

    # labels
    p.add_argument("--class_names", type=str, default="",
                   help="Comma-separated 8-class names, e.g. AD,FA,PT,TA,DC,LC,MC,PC")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
