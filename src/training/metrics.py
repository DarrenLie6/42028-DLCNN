from __future__ import annotations
import torch

""" Segmentation metrics for BRIGHT disaster assessment.

    - 4 classes: 0=Background, 1=Intact, 2=Damaged, 3=Destroyed
    - Background (index 0) is excluded from mean_iou and mean_f1.
    - Confusion matrix is accumulated across batches for exact results.
"""

NUM_CLASSES = 4
IGNORE_INDEX = 0
LABEL_NAMES = {0: "Background", 1: "Intact", 2: "Damaged", 3: "Destroyed"}
CLASS_WEIGHTS = [0.0, 1.0, 7.8, 13.0]

class SegmentationMetrics:
    """
    Accumulates a confusion matrix over batches, then computes per-class
    IoU and F1 for classes 1-3 
    Background is tracked but excluded from means.
    """
    
    def __init__(self, num_classes: int = NUM_CLASSES, ignore_index: int = IGNORE_INDEX,
                 device: torch.device | str = "cuda"):
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
        
    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate one batch into the confusion matrix."""
        
        preds = logits.argmax(dim=1).to(self.device)   # (B, H, W)
        targets = targets.to(self.device)

        # Discard background pixels entirely — they never touch the CM
        valid_mask = targets != self.ignore_index       # (B, H, W) bool
        preds_flat = preds[valid_mask]                  # (N,)
        targets_flat = targets[valid_mask]                # (N,)

        # bincount trick: encode (true, pred) as a single integer index
        indices = targets_flat * self.num_classes + preds_flat
        batch_cm = torch.bincount(
            indices, minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

        self.conf_matrix += batch_cm
        
    # def compute(self) -> dict[str, float]:
    #     """
    #     Compute per-class IoU & F1 from the accumulated confusion matrix.

    #     Returns flat dict:
    #         iou/Background, iou/Intact, iou/Damaged, iou/Destroyed
    #         f1/Background,  f1/Intact,  f1/Damaged,  f1/Destroyed
    #         mean_iou   — average of classes 1-3 only
    #         mean_f1    — average of classes 1-3 only
    #     """
    #     cm  = self.conf_matrix.float()      # (C, C)
    #     tp  = cm.diag()                     # (C,) true positives
    #     fp  = cm.sum(dim=0) - tp            # (C,) false positives
    #     fn  = cm.sum(dim=1) - tp            # (C,) false negatives
    #     eps = 1e-7

    #     iou = tp / (tp + fp + fn + eps)         # (C,)
    #     f1  = (2 * tp) / (2 * tp + fp + fn + eps)  # (C,)

    #     results: dict[str, float] = {}
    #     for idx, name in LABEL_NAMES.items():
    #         results[f"iou/{name}"] = iou[idx].item()
    #         results[f"f1/{name}"]  = f1[idx].item()

    #     # Mean over foreground only: classes 1, 2, 3
    #     fg = [i for i in LABEL_NAMES if i != self.ignore_index]
    #     results["mean_iou"] = iou[fg].mean().item()
    #     results["mean_f1"]  = f1[fg].mean().item()

    #     return results
    
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
        
    def compute(self) -> dict[str, float]:
        cm  = self.conf_matrix.float()
        tp  = cm.diag()
        fp  = cm.sum(dim=0) - tp
        fn  = cm.sum(dim=1) - tp
        eps = 1e-7

        iou = tp / (tp + fp + fn + eps)
        f1  = (2 * tp) / (2 * tp + fp + fn + eps)
        
        # Per-class accuracy = TP / (TP + FN) = true positive rate
        acc = tp / (cm.sum(dim=1) + eps)             # (C,)

        results: dict[str, float] = {}
        for idx, name in LABEL_NAMES.items():
            results[f"iou/{name}"] = iou[idx].item()
            results[f"f1/{name}"]  = f1[idx].item()
            results[f"acc/{name}"] = acc[idx].item()  # ← add this

        fg = [i for i in LABEL_NAMES if i != self.ignore_index]
        results["mean_iou"] = iou[fg].mean().item()
        results["mean_f1"]  = f1[fg].mean().item()
        results["mean_acc"] = acc[fg].mean().item()   # ← add this

        return results