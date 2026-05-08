from __future__ import annotations
import torch

""" Segmentation metrics for BRIGHT disaster assessment.

    - 5 classes: 0=Un-classified, 1=No-Damage, 2=Minor-Damage, 3=Major-Damage, 4=Destroyed
    - No class is ignored (all 5 classes included)
    - Confusion matrix is accumulated across batches for exact results.
"""

NUM_CLASSES = 5
IGNORE_INDEX = None
LABEL_NAMES = {0: "Un-classified", 1: "No-Damage", 2: "Minor-Damage", 3: "Major-Damage", 4: "Destroyed"}
CLASS_WEIGHTS = [0.5, 1.0, 2.0, 4.0, 4.0]  # Higher weight for destroyed class

class SegmentationMetrics:
    """
    Accumulates a confusion matrix over batches, then computes per-class
    IoU and F1 for classes 1-3 
    Background is tracked but excluded from means.
    """
    
    def __init__(self, num_classes: int = NUM_CLASSES, ignore_index: int = IGNORE_INDEX,
                 device: torch.device | str = "cpu"):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = torch.device(device)
        
        # rows = true class, cols = predicted class
        self.conf_matrix = torch.zeros(
            num_classes, num_classes, dtype=torch.long, device=self.device
        )
        
    # public API
    def reset(self) -> None:
        """Zero the confusion matrix. Call at start of every epoch"""
        self.conf_matrix.zero_()
        
    @torch.no_grad
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate one batch into the confusion matrix."""
        
        preds = logits.argmax(dim=1).to(self.device)   # (B, H, W)
        targets = targets.to(self.device)

        # Apply masking only if ignore_index is specified
        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index       # (B, H, W) bool
            preds_flat = preds[valid_mask]                  # (N,)
            targets_flat = targets[valid_mask]              # (N,)
        else:
            # Use all pixels if no ignore_index
            preds_flat = preds.reshape(-1)
            targets_flat = targets.reshape(-1)

        # bincount trick: encode (true, pred) as a single integer index
        indices = targets_flat * self.num_classes + preds_flat
        batch_cm = torch.bincount(
            indices, minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

        self.conf_matrix += batch_cm
        
    def compute(self) -> dict[str, float]:
        """
        Compute per-class IoU, F1, and Accuracy from the accumulated confusion matrix.

        Returns flat dict:
            iou/No-Damage, iou/Minor-Damage, iou/Major-Damage, iou/Destroyed, iou/Un-classified
            f1/No-Damage,  f1/Minor-Damage,  f1/Major-Damage,  f1/Destroyed,  f1/Un-classified
            acc/No-Damage, acc/Minor-Damage, acc/Major-Damage, acc/Destroyed, acc/Un-classified
            mean_iou   — average of all 5 classes
            mean_f1    — average of all 5 classes
            mean_acc   — average of all 5 classes
        """
        cm  = self.conf_matrix.float()      # (C, C)
        tp  = cm.diag()                     # (C,) true positives
        fp  = cm.sum(dim=0) - tp            # (C,) false positives
        fn  = cm.sum(dim=1) - tp            # (C,) false negatives
        eps = 1e-7

        iou = tp / (tp + fp + fn + eps)         # (C,)
        f1  = (2 * tp) / (2 * tp + fp + fn + eps)  # (C,)
        acc = tp / (cm.sum(dim=1) + eps)    # (C,) per-class accuracy (recall)

        results: dict[str, float] = {}
        for idx, name in LABEL_NAMES.items():
            results[f"iou/{name}"] = iou[idx].item()
            results[f"f1/{name}"]  = f1[idx].item()
            results[f"acc/{name}"] = acc[idx].item()

        # Mean over all classes (no class ignored)
        results["mean_iou"] = iou.mean().item()
        results["mean_f1"]  = f1.mean().item()
        results["mean_acc"] = acc.mean().item()

        return results
    
    def to(self, device: torch.device | str) -> "SegmentationMetrics":
        """Move internal CM tensor to device. Call after model.to(device)."""
        self.device = torch.device(device)
        self.conf_matrix = self.conf_matrix.to(self.device)
        
        return self
    
    def __repr__(self) -> str:
        return (
            f"SegmentationMetrics(num_classes={self.num_classes}, "
            f"ignore_index={self.ignore_index}, device={self.device})"
        )