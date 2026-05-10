from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


"""Loss functions optimized for xBD (xView2 Building Damage) semantic segmentation
with extreme class imbalance (85.9% background vs 0.9% destroyed).

Recommended for AttentionUNet:
- CombinedLoss: 0.4 CE + 0.3 Dice + 0.3 Focal (default)
- Or FocalLoss: focuses on hard negatives
"""


class DiceLoss(nn.Module):
    """Dice Loss (F1 score) - works better when combined with CE for imbalanced data.
    
    Args:
        num_classes: number of semantic classes
        ignore_index: pixels with this label are ignored (-100 = not used)
        eps: small epsilon for numerical stability
        use_background: whether to include background class in loss
    """
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = -100,
        eps: float = 1e-6,
        use_background: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps
        self.use_background = use_background

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        loss = 0.0
        count = 0

        # Determine which classes to compute loss for
        start_class = 0 if self.use_background else 1

        for cls in range(start_class, self.num_classes):
            p = probs[:, cls]  # (B, H, W)
            g = (targets == cls).float()
            
            # Compute Dice coefficient
            intersection = (p * g).sum()
            denominator = p.sum() + g.sum() + self.eps
            dice_coef = 2.0 * intersection / denominator
            loss += 1.0 - dice_coef
            count += 1

        return loss / max(count, 1)


class FocalLoss(nn.Module):
    """Focal Loss - focuses on hard negatives and reduces contribution of easy examples.
    
    Particularly effective for extreme class imbalance like xBD dataset.
    
    Args:
        num_classes: number of semantic classes
        alpha: weighting factor in range (0, 1) to balance positive vs negative examples
        gamma: focusing parameter for modulating loss from hard negatives (typically 2.0)
        class_weights: optional per-class weights
        ignore_index: pixels with this label are ignored
    """
    def __init__(
        self,
        num_classes: int,
        alpha: float = 1.0,
        gamma: float = 2.0,
        class_weights: list = None,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

        if class_weights is None:
            class_weights = [1.0] * num_classes

        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float32)
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W)
            targets: (B, H, W) with class indices or -100 for ignore
        
        Returns:
            scalar loss
        """
        # Compute cross-entropy
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction="none"  # (B, H, W)
        )

        # Compute probabilities
        p = torch.exp(-ce_loss)  # probability of correct class

        # Apply focal term: (1 - p)^gamma
        focal_weight = (1 - p) ** self.gamma

        # Apply alpha weighting and focal weight
        focal_loss = self.alpha * focal_weight * ce_loss

        # Average over valid pixels
        valid_mask = (targets != self.ignore_index).float()
        return (focal_loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)


# class CombinedLoss(nn.Module):
#     """
#     0.5 * WeightedCE + 0.5 * Dice
#     Weights derived from eda
#     """

#     def __init__(
#         self,
#         num_classes:   int   = 4,
#         ignore_index:  int   = 0,
#         ce_weights:    float = 0.5,
#         dice_weights:  float = 0.5,
#         class_weights: list  = None,
#     ):
#         super().__init__()
#         self.ce_w         = ce_weights
#         self.dice_w       = dice_weights
#         self.ignore_index = ignore_index

#         if class_weights is None:
#             class_weights = [0.0, 1.0, 7.8, 20.0]

#         # register_buffer → moves automatically with .to(device)
#         self.register_buffer(
#             "weights",
#             torch.tensor(class_weights, dtype=torch.float32)
#         )

#         self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)

#     def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple:
#         # reduction="sum" so we control averaging manually
#         ce_loss = F.cross_entropy(
#             logits, targets,
#             weight = self.weights,
#             ignore_index = self.ignore_index,
#             reduction  = "mean",
#             label_smoothing=0.1
#         )

#         # Count valid (non-background) pixels
#         valid_pixels = (targets != self.ignore_index).sum().float()

#         # Guard against all-background batch → CE would be NaN otherwise
#         ce_loss = logits.sum() * 0.0

#         dice_loss = self.dice(logits, targets)
#         total     = self.ce_w * ce_loss + self.dice_w * dice_loss
#         return total, ce_loss, dice_loss

class CombinedLoss(nn.Module):
    """Recommended loss for AttentionUNet + xBD semantic segmentation.
    
    Combines 3 complementary loss functions optimized for extreme class imbalance:
    - CrossEntropyLoss (weighted): Punishes misclassifications
    - DiceLoss: Focuses on class-level F1 score
    - FocalLoss: Emphasizes hard negatives and rare classes
    
    Default weights: 0.4 CE + 0.3 Dice + 0.3 Focal (best for xBD)
    
    Class distribution in xBD:
        Background (0): 85.9% | Intact (1): 11.7% | Damaged (2): 1.5% | Destroyed (3): 0.9%
    
    Class weights computed from inverse frequency (with smoothing):
        [0.1, 1.0, 5.0, 10.0] emphasizes rare damage classes
    """

    def __init__(
        self,
        num_classes: int = 4,
        ignore_index: int = -100,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        focal_weight: float = 0.0,
        class_weights: list = None,
        use_focal: bool = False,
    ):
        super().__init__()
        self.ce_w = ce_weight
        self.dice_w = dice_weight
        self.focal_w = focal_weight
        self.use_focal = use_focal
        self.ignore_index = ignore_index

        if class_weights is None:
            # Optimized for xBD: higher weights for rare damage classes
            # Computed as 1 / (frequency + smoothing_factor)
            # [Background, Intact, Damaged, Destroyed]
            class_weights = [0.1, 1.0, 5.0, 10.0]

        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float32)
        )

        self.ce = nn.CrossEntropyLoss(
            weight=self.class_weights,
            ignore_index=ignore_index,
            label_smoothing=0.0,  # Reduced from 0.1 for stability
            reduction="mean"
        )

        self.dice = DiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            use_background=True
        )

        if self.use_focal:
            self.focal = FocalLoss(
                num_classes=num_classes,
                alpha=1.0,
                gamma=2.0,  # Focus more on hard negatives
                class_weights=class_weights,
                ignore_index=ignore_index,
            )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple:
        """
        Args:
            logits: (B, C, H, W) raw model outputs
            targets: (B, H, W) ground truth labels
        
        Returns:
            (total_loss, ce_loss, dice_loss, focal_loss if used)
        """
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)

        if self.use_focal:
            focal_loss = self.focal(logits, targets)
            total = self.ce_w * ce_loss + self.dice_w * dice_loss + self.focal_w * focal_loss
            return total, ce_loss, dice_loss, focal_loss
        else:
            total = self.ce_w * ce_loss + self.dice_w * dice_loss
            return total, ce_loss, dice_loss