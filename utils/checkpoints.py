import torch
from typing import Tuple, List

def load_checkpoint(model, ckpt_path: str, device: str = "cpu"):
    obj = torch.load(ckpt_path, map_location=device)
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    missing, unexpected = model.load_state_dict(state, strict=False)
    return missing, unexpected
