# augmentation.py
import random
from PIL import Image, ImageEnhance

def random_augment(img):
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    if random.random() < 0.3:
        angle = random.uniform(-15, 15)
        img = img.rotate(angle)
    if random.random() < 0.3:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    return img
