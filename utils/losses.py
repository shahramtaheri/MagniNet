import torch
import torch.nn as nn

def dual_head_loss(logits_mc, y_mc, logits_bin, y_bin, alpha: float = 0.6):
    ce = nn.CrossEntropyLoss()(logits_mc, y_mc)
    bce = nn.BCEWithLogitsLoss()(logits_bin, y_bin)
    return alpha * ce + (1 - alpha) * bce, {"loss_mc": ce.item(), "loss_bin": bce.item()}
