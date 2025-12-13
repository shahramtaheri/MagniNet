# train_magninet.py
# Full training loop for MagniNet: AMP + AdamW + CosineAnnealing + dual-head loss + checkpoints
# pip install timm

import os
import json
import time
import math
import random
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from PIL import Image

try:
    import timm
except ImportError as e:
    raise ImportError("Please install timm: pip install timm") from e


# -------------------------
# Reproducibility
# -------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # faster
    torch.backends.cudnn.benchmark = True       # faster


# -------------------------
# Simple image dataset (folder structure)
# -------------------------
class ImageFolderWithLabels(Dataset):
    """
    Expects:
      root/
        images/
          <id>.png / <id>.jpg ...
        labels.json  (maps filename -> integer class 0..7)
        binary_labels.json (optional; if missing, derived from multiclass using mapping)
    """
    def __init__(self, root: str, image_size: int = 224, augment: bool = False):
        self.root = root
        self.image_dir = os.path.join(root, "images")
        self.image_size = image_size
        self.augment = augment

        labels_path = os.path.join(root, "labels.json")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Missing labels.json at: {labels_path}")

        with open(labels_path, "r") as f:
            self.labels = json.load(f)

        bin_path = os.path.join(root, "binary_labels.json")
        self.has_binary = os.path.exists(bin_path)
        if self.has_binary:
            with open(bin_path, "r") as f:
                self.bin_labels = json.load(f)
        else:
            self.bin_labels = {}

        self.files = sorted(list(self.labels.keys()))
        if len(self.files) == 0:
            raise RuntimeError(f"No files found in labels.json at: {root}")

    @staticmethod
    def _resize(img: Image.Image, size: int) -> Image.Image:
        return img.resize((size, size), resample=Image.BILINEAR)

    @staticmethod
    def _to_tensor(img: Image.Image) -> torch.Tensor:
        # [0,1] float tensor CHW
        x = torch.from_numpy(
            (torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
             .view(img.size[1], img.size[0], len(img.getbands()))
             .numpy())
        ).float() / 255.0
        x = x.permute(2, 0, 1).contiguous()
        return x

    def _augment(self, img: Image.Image) -> Image.Image:
        # Lightweight augmentations (you can replace with torchvision/albumentations)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        # small rotation
        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            img = img.rotate(angle, resample=Image.BILINEAR)
        return img

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        fname = self.files[idx]
        path = os.path.join(self.image_dir, fname)
        img = Image.open(path).convert("RGB")

        if self.augment:
            img = self._augment(img)

        img = self._resize(img, self.image_size)
        x = self._to_tensor(img)

        y_mc = int(self.labels[fname])

        if self.has_binary:
            y_bin = int(self.bin_labels[fname])
        else:
            # Derive benign/malignant from 8-class by convention:
            # benign: AD, FA, PT, TA ; malignant: DC, LC, MC, PC
            # Here we assume your class indexing follows that order.
            # If your indices differ, change this mapping.
            y_bin = 0 if y_mc in [0, 1, 2, 3] else 1

        return x, torch.tensor(y_mc, dtype=torch.long), torch.tensor(y_bin, dtype=torch.float32)


# -------------------------
# MagniNet model (same as before, included for completeness)
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
        g = self.mlp(x)  # [B,N,1]
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
    def __init__(self, dim: int, num_heads: int = 8, attn_drop: float = 0.0, drop: float = 0.15, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True)
        self.drop1 = nn.Dropout(drop)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FFN(dim, mlp_ratio=mlp_ratio, drop=drop)
        self.gate = AdaptiveGate(dim)

    def forward(self, x):
        x1 = self.norm1(x)
        attn_out, _ = self.attn(x1, x1, x1, need_weights=False)
        x = x + self.drop1(attn_out)
        x = x + self.ffn(self.norm2(x))
        x = self.gate(x)
        return x


class EfficientNetLocalEncoder(nn.Module):
    def __init__(self, model_name: str = "efficientnet_b0", out_dim: int = 256, pretrained: bool = True):
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

        # stable fusion projection
        self.fuse = nn.Conv2d(out_dim * len(feat_channels), out_dim, kernel_size=1, bias=False)

    def forward(self, x):
        feats = self.backbone(x)  # list of [B,C,H,W]
        proj_feats = []
        for f, p, bn in zip(feats, self.proj, self.bn):
            y = bn(p(f))
            proj_feats.append(y)

        Hmax = max(t.shape[-2] for t in proj_feats)
        Wmax = max(t.shape[-1] for t in proj_feats)

        aligned = []
        for t in proj_feats:
            if t.shape[-2:] != (Hmax, Wmax):
                t = F.interpolate(t, size=(Hmax, Wmax), mode="bilinear", align_corners=False)
            aligned.append(t)

        fused = self.fuse(torch.cat(aligned, dim=1))  # [B,D,H,W]
        B, D, H, W = fused.shape
        tokens = fused.flatten(2).transpose(1, 2)     # [B, H*W, D]
        return tokens


class MagniNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        dim: int = 256,
        num_heads: int = 8,
        pretrained_backbones: bool = True,
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
        local_tokens = self.local_encoder(x)                # [B,Nl,D]
        swin_feat = self.swin(x)[0]                         # [B,C,H,W]
        swin_feat = self.swin_proj(swin_feat)               # [B,D,H,W]
        global_tokens = swin_feat.flatten(2).transpose(1, 2)  # [B,Ng,D]

        fused = self.cross_attn(global_tokens, local_tokens)  # [B,Ng,D]
        fused = self.blocks(fused)
        fused = self.norm(fused)

        emb = fused.mean(dim=1)                             # [B,D]
        logits_mc = self.head_multiclass(emb)               # [B,8]
        logits_bin = self.head_binary(emb).squeeze(-1)      # [B]
        return logits_mc, logits_bin


# -------------------------
# Metrics
# -------------------------
@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()

@torch.no_grad()
def binary_accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = (torch.sigmoid(logits) >= 0.5).float()
    return (pred == y).float().mean().item()


# -------------------------
# Config
# -------------------------
@dataclass
class TrainConfig:
    # data
    data_root: str                      # e.g., "/path/BreakHis/40x" (contains images/ + labels.json)
    image_size: int = 224
    num_classes: int = 8

    # training
    epochs: int = 40
    batch_size: int = 32
    num_workers: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-2
    alpha: float = 0.6                  # loss = alpha*L_multi + (1-alpha)*L_binary  (paper uses 0.6)
    drop: float = 0.15
    amp: bool = True
    seed: int = 42

    # model
    dim: int = 256
    heads: int = 8
    swin_name: str = "swin_tiny_patch4_window7_224"
    pretrained_backbones: bool = True

    # scheduler
    cosine_tmax: Optional[int] = None   # if None -> epochs
    min_lr: float = 1e-6

    # checkpoint
    out_dir: str = "./runs/magninet"
    save_best_only: bool = True


# -------------------------
# Trainer
# -------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    cfg: TrainConfig,
    device: str,
    ce_loss: nn.Module,
    bce_loss: nn.Module,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_acc_mc = 0.0
    total_acc_bin = 0.0
    n = 0

    for x, y_mc, y_bin in loader:
        x = x.to(device, non_blocking=True)
        y_mc = y_mc.to(device, non_blocking=True)
        y_bin = y_bin.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=cfg.amp):
            logits_mc, logits_bin = model(x)
            loss_mc = ce_loss(logits_mc, y_mc)
            loss_bin = bce_loss(logits_bin, y_bin)
            loss = cfg.alpha * loss_mc + (1.0 - cfg.alpha) * loss_bin

        if cfg.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc_mc += accuracy_from_logits(logits_mc, y_mc) * bs
        total_acc_bin += binary_accuracy_from_logits(logits_bin, y_bin) * bs
        n += bs

    return {
        "loss": total_loss / max(1, n),
        "acc_mc": total_acc_mc / max(1, n),
        "acc_bin": total_acc_bin / max(1, n),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    cfg: TrainConfig,
    device: str,
    ce_loss: nn.Module,
    bce_loss: nn.Module,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc_mc = 0.0
    total_acc_bin = 0.0
    n = 0

    for x, y_mc, y_bin in loader:
        x = x.to(device, non_blocking=True)
        y_mc_
