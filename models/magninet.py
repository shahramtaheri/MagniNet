
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, global_tokens: torch.Tensor, local_tokens: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        self.fuse = nn.Conv2d(out_dim * len(feat_channels), out_dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor):
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
