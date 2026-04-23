from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


"""Combined loss: weighted CrossEntropy + Dice
    - skips the background index (0)
    - class_weights to handle severe imbalance
"""


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_index: int = 0, eps: float = 1e-6):
        super().__init__()
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.eps          = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        loss  = 0.0
        count = 0

        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue
            p    = probs[:, cls]                              # (B, H, W)
            g    = (targets == cls).float()
            mask = (targets != self.ignore_index).float()
            p, g = p * mask, g * mask
            intersection = (p * g).sum()
            denominator  = p.sum() + g.sum() + self.eps
            loss  += 1.0 - 2.0 * intersection / denominator
            count += 1

        return loss / count


class CombinedLoss(nn.Module):
    """
    0.5 * WeightedCE + 0.5 * Dice
    Weights derived from eda
    """

    def __init__(
        self,
        num_classes:   int   = 4,
        ignore_index:  int   = 0,
        ce_weights:    float = 0.5,
        dice_weights:  float = 0.5,
        class_weights: list  = None,
    ):
        super().__init__()
        self.ce_w         = ce_weights
        self.dice_w       = dice_weights
        self.ignore_index = ignore_index

        if class_weights is None:
            class_weights = [0.0, 1.0, 7.8, 20.0]

        # register_buffer → moves automatically with .to(device)
        self.register_buffer(
            "weights",
            torch.tensor(class_weights, dtype=torch.float32)
        )

        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple:
        # reduction="sum" so we control averaging manually
        ce_loss = F.cross_entropy(
            logits, targets,
            weight = self.weights,
            ignore_index = self.ignore_index,
            reduction  = "mean",
            label_smoothing=0.1
        )

        # Count valid (non-background) pixels
        valid_pixels = (targets != self.ignore_index).sum().float()

        # Guard against all-background batch → CE would be NaN otherwise
        if valid_pixels > 0:
            ce_loss = ce_loss / valid_pixels
        else:
            ce_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

        dice_loss = self.dice(logits, targets)
        total     = self.ce_w * ce_loss + self.dice_w * dice_loss
        return total, ce_loss, dice_loss