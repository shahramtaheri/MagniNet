import os, json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .label_mapping import multiclass_to_binary

class BreakHisFolderDataset(Dataset):
    def __init__(self, root: str, image_size: int = 224, augment_fn=None):
        self.root = root
        self.image_dir = os.path.join(root, "images")
        labels_path = os.path.join(root, "labels.json")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Missing labels.json at {labels_path}")
        with open(labels_path, "r") as f:
            self.labels = json.load(f)

        bin_path = os.path.join(root, "binary_labels.json")
        self.has_binary = os.path.exists(bin_path)
        self.bin_labels = {}
        if self.has_binary:
            with open(bin_path, "r") as f:
                self.bin_labels = json.load(f)

        self.files = sorted(self.labels.keys())
        self.image_size = image_size
        self.augment_fn = augment_fn

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        path = os.path.join(self.image_dir, fname)
        img = Image.open(path).convert("RGB")
        if self.augment_fn is not None:
            img = self.augment_fn(img)
        img = img.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        y_mc = int(self.labels[fname])
        if self.has_binary:
            y_bin = int(self.bin_labels[fname])
        else:
            y_bin = multiclass_to_binary(y_mc)

        return x, torch.tensor(y_mc, dtype=torch.long), torch.tensor(y_bin, dtype=torch.float32), fname
