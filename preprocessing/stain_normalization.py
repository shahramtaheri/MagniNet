# stain_normalization.py
import cv2
import numpy as np

def reinhard_normalization(src_img, target_img):
    src = cv2.cvtColor(src_img, cv2.COLOR_RGB2LAB).astype(np.float32)
    tgt = cv2.cvtColor(target_img, cv2.COLOR_RGB2LAB).astype(np.float32)

    for i in range(3):
        src[:, :, i] = (
            (src[:, :, i] - src[:, :, i].mean()) /
            (src[:, :, i].std() + 1e-8)
        ) * tgt[:, :, i].std() + tgt[:, :, i].mean()

    out = cv2.cvtColor(src.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return out
