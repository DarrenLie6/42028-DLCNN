"""
Semantic segmentation model for xView2 building damage detection.

Outputs per-pixel damage classification:
  - 0: Background
  - 1: Intact
  - 2: Damaged
  - 3: Destroyed
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)
from typing import Dict


class DeepLabV3Model(nn.Module):
    """
    DeepLabV3 with ResNet-50 backbone for semantic segmentation.

    Outputs: (B, C, H, W) logits where C = num_classes
    """

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
    ):
        """
        Args:
            num_classes: 4 (Background, Intact, Damaged, Destroyed)
            pretrained:  Load COCO pretrained weights (backbone + head)
        """
        super().__init__()

        self.num_classes = num_classes

        # Load DeepLabV3 with updated weights API
        weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        self.model = deeplabv3_resnet50(weights=weights)

        # Replace classifier head for custom number of classes
        self.model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)

        # Replace auxiliary head (present in both train and eval)
        if self.model.aux_classifier is not None:
            self.model.aux_classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (B, 3, H, W) input images

        Returns (training):
            dict with 'out' (main output) and 'aux' (auxiliary output)

        Returns (inference / eval mode):
            dict with 'out' — (B, num_classes, H, W) logits
        """
        return self.model(x)

    def set_inference_mode(self, mode: bool = True) -> None:
        """Set model to inference/training mode."""
        if mode:
            self.model.eval()
        else:
            self.model.train()

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze/unfreeze the ResNet-50 backbone."""
        for param in self.model.backbone.parameters():
            param.requires_grad = not freeze


def build_semantic_model(
    num_classes: int = 4,
    pretrained: bool = True,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> DeepLabV3Model:
    """
    Build and return a DeepLabV3 semantic segmentation model.

    Args:
        num_classes: Number of output classes (4 for xView2)
        pretrained:  Load pretrained weights via new torchvision weights API
        device:      Device to place model on

    Returns:
        DeepLabV3Model on device
    """
    model = DeepLabV3Model(
        num_classes=num_classes,
        pretrained=pretrained,
    )
    model = model.to(device)
    return model