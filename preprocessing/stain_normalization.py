# stain_normalization.py
# Reinhard stain normalization for histopathology images

import cv2
import numpy as np


def reinhard_normalization(src_img: np.ndarray, target_img: np.ndarray) -> np.ndarray:
    """
    Perform Reinhard stain normalization.

    Args:
        src_img: Source RGB image (H,W,3), uint8
        target_img: Target RGB image (H,W,3), uint8

    Returns:
        Normalized RGB image (H,W,3), uint8
    """

    src = cv2.cvtColor(src_img, cv2.COLOR_RGB2LAB).astype(np.float32)
    tgt = cv2.cvtColor(target_img, cv2.COLOR_RGB2LAB).astype(np.float32)

    for c in range(3):
        src_mean, src_std = src[:, :, c].mean(), src[:, :, c].std()
        tgt_mean, tgt_std = tgt[:, :, c].mean(), tgt[:, :, c].std()
        src[:, :, c] = (src[:, :, c] - src_mean) / (src_std + 1e-8)
        src[:, :, c] = src[:, :, c] * tgt_std + tgt_mean

    out = np.clip(src, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_LAB2RGB)
    return out
