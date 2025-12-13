# evaluate_external.py
# External validation script for MagniNet.
#
# Supports external datasets with:
#   - binary labels only (recommended for BACH/Camelyon16/PCam/TCGA-BRCA),
#   - or both multi-class and binary (if you provide labels.json with 0..7 and binary_labels.json).
#
# Folder structure (one dataset at a time):
#   external_root/
#     images/
#     binary_labels.json        # REQUIRED for external eval: filename -> 0/1
#     labels.json               # OPTIONAL: filename -> 0..7 (if available)
#
# Output:
#   out_dir/<dataset_name>/
#     external_report.json
#     confusion_binary.png
#     confusion_multiclass.png  (only if labels.json exists)
#
# pip install timm scikit-learn matplotlib

import os
import json
import argparse
from typing import Optional, List, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

try:
    import timm
except ImportError as e:
    raise ImportError("Please install timm: pip install timm") from e


# -------------------------
# Dataset
# -------------------------
class ExternalImageDataset(Dataset):
    """
    External dataset loader.

    Required:
      binary_labels.json : { "img.jpg": 0/1, ... }

    Optional:
      labels.json        : { "img.jpg": 0..7, ... }   (if you want 8-class evaluation too)
    """
    def __init__(self, root: str, image_size: int = 224):
        self.root = root
        self.image_dir = os.path.join(root, "images")
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Missing images/ folder: {self.image_dir}")

        bin_path = os.path.join(root, "binary_labels.json")
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Missing binary_labels.json at: {bin_path}")

        with open(bin_path, "r") as f:
            self.bin_labels = json.load(f)

        self.labels_path = os.path.join(root, "labels.json")
        self.has_multiclass = os.path.exists(self.labels_path)
        self.mc_labels = {}
        if self.has_multiclass:
            with open(self.labels_path, "r") as f:
                self.mc_labels = json.load(f)

        self.files = sorted(list(self.bin_labels.keys()))
        if len(self.files) == 0:
            raise RuntimeError(f"No entries in binary_labels.json at: {bin_path}")

        self.image_size = image_size

    @staticmethod
    def _resize(img: Image.Image, size: int) -> Image.Image:
        return img.resize((size, size), resample=Image.BILINEAR)

    @staticmethod
    def _to_tensor(img: Image.Image) -> torch.Tensor:
        arr = np.array(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return x

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        fname = self.files[idx]
        img_path = os.path.join(self.image_dir, fname)
        img = Image.open(img_path).convert("RGB")
        img = self._resize(img, self.image_size)
        x = self._to_tensor(img)

        y_bin = int(self.bin_labels[fname])
        if self.has_multiclass:
            y_mc = int(self.mc_labels[fname])
        else:
            y_mc = -1  # unknown

        return x, torch.tensor(y_mc, dtype=torch.long), torch.tensor(y_bin, dtype=torch.float32), fname


# -------------------------
# MagniNet model (same architecture as train/test scripts)
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
# Utilities
# -------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

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

def save_cm_png(cm: np.ndarray, class_names: List[str], out_path: str, title: str):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]}",
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def evaluate_external(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    out_dir = os.path.join(args.out_dir, args.dataset_name)
    ensure_dir(out_dir)

    ds = ExternalImageDataset(args.external_root, image_size=args.image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

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

    y_bin_true, y_bin_pred, y_bin_prob = [], [], []
    y_mc_true, y_mc_pred = [], []
    filenames = []

    for x, y_mc, y_bin, fn in loader:
        x = x.to(device, non_blocking=True)
        logits_mc, logits_bin = model(x)

        prob_bin = torch.sigmoid(logits_bin).detach().cpu().numpy()
        pred_bin = (prob_bin >= 0.5).astype(np.int64)

        y_bin_true.extend(y_bin.numpy().astype(np.int64).tolist())
        y_bin_pred.extend(pred_bin.tolist())
        y_bin_prob.extend(prob_bin.tolist())

        if ds.has_multiclass:
            pred_mc = logits_mc.argmax(dim=1).detach().cpu().numpy()
            y_mc_true.extend(y_mc.numpy().astype(np.int64).tolist())
            y_mc_pred.extend(pred_mc.tolist())

        filenames.extend(list(fn))

    y_bin_true = np.array(y_bin_true, dtype=np.int64)
    y_bin_pred = np.array(y_bin_pred, dtype=np.int64)
    y_bin_prob = np.array(y_bin_prob, dtype=np.float32)

    # ---- Binary metrics (primary for external validation)
    bin_acc = accuracy_score(y_bin_true, y_bin_pred)
    bin_prec, bin_rec, bin_f1, _ = precision_recall_fscore_support(
        y_bin_true, y_bin_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_bin_true, y_bin_pred, labels=[0, 1]).ravel()
    bin_spec = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    try:
        bin_auc = roc_auc_score(y_bin_true, y_bin_prob)
    except ValueError:
        bin_auc = float("nan")

    bin_cm = confusion_matrix(y_bin_true, y_bin_pred, labels=[0, 1])
    save_cm_png(bin_cm, ["Benign", "Malignant"], os.path.join(out_dir, "confusion_binary.png"),
                f"{args.dataset_name} - Binary Confusion Matrix")

    report = {
        "dataset_name": args.dataset_name,
        "external_root": os.path.abspath(args.external_root),
        "checkpoint": os.path.abspath(args.checkpoint),
        "device": device,
        "binary": {
            "accuracy": float(bin_acc),
            "precision": float(bin_prec),
            "recall": float(bin_rec),
            "specificity": float(bin_spec),
            "f1": float(bin_f1),
            "auroc": float(bin_auc) if not (isinstance(bin_auc, float) and np.isnan(bin_auc)) else None,
            "confusion_matrix": bin_cm.tolist(),
            "tn_fp_fn_tp": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        },
        "multiclass": None,
        "notes": {
            "multiclass_evaluated": bool(ds.has_multiclass),
            "binary_labels_required": True,
        }
    }

    # ---- Optional multi-class metrics (only if labels.json is present)
    if ds.has_multiclass:
        y_mc_true = np.array(y_mc_true, dtype=np.int64)
        y_mc_pred = np.array(y_mc_pred, dtype=np.int64)

        mc_acc = accuracy_score(y_mc_true, y_mc_pred)
        mc_prec, mc_rec, mc_f1, _ = precision_recall_fscore_support(
            y_mc_true, y_mc_pred, average="macro", zero_division=0
        )
        mc_cm = confusion_matrix(y_mc_true, y_mc_pred, labels=list(range(args.num_classes)))

        class_names = args.class_names.split(",") if args.class_names else ["AD", "FA", "PT", "TA", "DC", "LC", "MC", "PC"]
        if len(class_names) != args.num_classes:
            class_names = [str(i) for i in range(args.num_classes)]

        save_cm_png(mc_cm, class_names, os.path.join(out_dir, "confusion_multiclass.png"),
                    f"{args.dataset_name} - Multi-class Confusion Matrix")

        report["multiclass"] = {
            "accuracy": float(mc_acc),
            "macro_precision": float(mc_prec),
            "macro_recall": float(mc_rec),
            "macro_f1": float(mc_f1),
            "confusion_matrix": mc_cm.tolist(),
        }

    with open(os.path.join(out_dir, "external_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Print a compact summary
    print(f"\n=== External Validation: {args.dataset_name} ===")
    print("Binary (primary):")
    print(f"  Accuracy   : {bin_acc:.4f}")
    print(f"  Precision  : {bin_prec:.4f}")
    print(f"  Recall     : {bin_rec:.4f}")
    print(f"  Specificity: {bin_spec:.4f}")
    print(f"  F1         : {bin_f1:.4f}")
    print(f"  AUROC      : {bin_auc:.4f}" if report["binary"]["auroc"] is not None else "  AUROC      : n/a")

    if report["multiclass"] is not None:
        print("Multi-class (optional):")
        print(f"  Accuracy        : {report['multiclass']['accuracy']:.4f}")
        print(f"  Macro Precision : {report['multiclass']['macro_precision']:.4f}")
        print(f"  Macro Recall    : {report['multiclass']['macro_recall']:.4f}")
        print(f"  Macro F1        : {report['multiclass']['macro_f1']:.4f}")

    print(f"\nSaved to: {os.path.abspath(out_dir)}")
    print(" - external_report.json")
    print(" - confusion_binary.png")
    if report["multiclass"] is not None:
        print(" - confusion_multiclass.png")


def parse_args():
    p = argparse.ArgumentParser(description="External validation for MagniNet (binary primary, optional 8-class).")

    p.add_argument("--dataset_name", type=str, required=True,
                   help="Name used for output folder, e.g., BACH or Camelyon16 or PatchCamelyon or TCGA_BRCA.")
    p.add_argument("--external_root", type=str, required=True,
                   help="Path to the external dataset folder containing images/ and binary_labels.json (and optional labels.json).")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to checkpoint (best.pt / last.pt / state_dict).")
    p.add_argument("--out_dir", type=str, default="./results/external",
                   help="Output base directory.")
    p.add_argument("--cpu", action="store_true", help="Force CPU evaluation.")

    # model params (must match training)
    p.add_argument("--num_classes", type=int, default=8)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--swin_name", type=str, default="swin_tiny_patch4_window7_224")
    p.add_argument("--drop", type=float, default=0.15)

    # loader
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)

    # optional names
    p.add_argument("--class_names", type=str, default="",
                   help="Comma-separated 8-class names, e.g. AD,FA,PT,TA,DC,LC,MC,PC (used only if labels.json exists).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_external(args)
