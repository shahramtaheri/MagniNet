# augmentation.py
# Data augmentation utilities for histopathology images

import random
from PIL import Image, ImageEnhance, ImageFilter


def random_augment(img: Image.Image) -> Image.Image:
    """
    Apply random augmentations to a PIL image.
    Augmentations are intentionally lightweight to
    preserve histopathological structures.
    """

    # Horizontal flip
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Vertical flip
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # Small rotation
    if random.random() < 0.3:
        angle = random.uniform(-15, 15)
        img = img.rotate(angle, resample=Image.BILINEAR)

    # Brightness jitter
    if random.random() < 0.3:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.85, 1.15))

    # Contrast jitter
    if random.random() < 0.3:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.85, 1.15))

    return img

