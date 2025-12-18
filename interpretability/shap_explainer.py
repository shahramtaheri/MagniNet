# interpretability/shap_explainer.py
# SHAP GradientExplainer utilities for MagniNet (binary or multiclass).

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class _MagniNetBinaryWrapper(nn.Module):
    """
    Wraps MagniNet forward to return a single scalar per sample (binary logit).
    SHAP can explain logits; you can convert to probability afterwards.
    """
    def __init__(self, magninet: nn.Module):
        super().__init__()
        self.m = magninet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, logits_bin = self.m(x)
        return logits_bin.unsqueeze(1)  # [B,1]


class _MagniNetMultiClassWrapper(nn.Module):
    """
    Wraps MagniNet forward to return logits for one chosen class index (scalar).
    """
    def __init__(self, magninet: nn.Module, class_index: int):
        super().__init__()
        self.m = magninet
        self.k = int(class_index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits_mc, _ = self.m(x)
        return logits_mc[:, self.k].unsqueeze(1)  # [B,1]


def build_shap_gradient_explainer(
    model: nn.Module,
    background: torch.Tensor,
    mode: str = "binary",
    class_index: int | None = None,
    device: str | torch.device = "cuda",
):
    """
    Build a SHAP GradientExplainer for MagniNet.

    Args:
        model: MagniNet model
        background: background tensor [N,3,H,W], small (e.g., 8-32 images), float in [0,1]
        mode: "binary" or "multiclass"
        class_index: required for multiclass if you want a specific class;
                     if None, you can decide class per image by building a new explainer.
        device: cuda/cpu

    Returns:
        explainer: shap.GradientExplainer
        wrapped_model: torch.nn.Module used inside explainer
    """
    import shap

    device = torch.device(device)
    model.eval().to(device)
    background = background.to(device)

    if mode.lower() == "binary":
        wrapped = _MagniNetBinaryWrapper(model).to(device)
    elif mode.lower() == "multiclass":
        if class_index is None:
            raise ValueError("class_index must be provided for multiclass SHAP wrapper.")
        wrapped = _MagniNetMultiClassWrapper(model, class_index=class_index).to(device)
    else:
        raise ValueError("mode must be 'binary' or 'multiclass'.")

    # GradientExplainer expects (model, background)
    explainer = shap.GradientExplainer(wrapped, background)
    return explainer, wrapped


def shap_values_for_images(
    explainer,
    images: torch.Tensor,
    device: str | torch.device = "cuda",
):
    """
    Compute SHAP values for a batch of images.

    Args:
        explainer: shap.GradientExplainer returned by build_shap_gradient_explainer
        images: [B,3,H,W] float in [0,1]

    Returns:
        shap_vals: numpy array with shape [B,3,H,W] (typically)
    """
    device = torch.device(device)
    imgs = images.to(device)

    # GradientExplainer returns a list for each output; we wrapped to 1 output => list length 1
    shap_vals = explainer.shap_values(imgs)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    return shap_vals


def shap_attribution_map(shap_vals: np.ndarray, reduce_channels: str = "sum_abs") -> np.ndarray:
    """
    Convert SHAP values [B,3,H,W] into a single heatmap [B,H,W].

    reduce_channels:
      - "sum_abs": sum over channels of absolute SHAP
      - "sum": sum over channels (signed)
      - "mean_abs": mean absolute
    """
    if shap_vals.ndim != 4:
        raise ValueError("Expected shap_vals with shape [B,3,H,W].")

    if reduce_channels == "sum_abs":
        m = np.sum(np.abs(shap_vals), axis=1)
    elif reduce_channels == "sum":
        m = np.sum(shap_vals, axis=1)
    elif reduce_channels == "mean_abs":
        m = np.mean(np.abs(shap_vals), axis=1)
    else:
        raise ValueError("Unsupported reduce_channels.")
    return m


def normalize_01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32)
    mn = x.min(axis=(-2, -1), keepdims=True)
    mx = x.max(axis=(-2, -1), keepdims=True)
    return (x - mn) / (mx - mn + eps)
