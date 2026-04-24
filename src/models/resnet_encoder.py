from __future__ import annotations
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, resnet34, ResNet34_Weights

class ResNetEncoder(nn.Module):
    """
    Shared ResNet encoder with separate input stems for optical and SAR.
     - ResNet-50: out_channels = [64, 256, 512, 1024, 2048]  (bottleneck blocks)
     - ResNet-34: out_channels = [64,  64, 128,  256,  512]  (basic blocks)
    """
    
    def __init__(self, pretrained: bool = True, backbone: str = "resnet50"):
        super().__init__()

        # Load backbone — use `base` to avoid conflict with `backbone` string param
        if backbone == "resnet50":
            base = resnet50(
                weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            )
            self.out_channels = [64, 256, 512, 1024, 2048]

        elif backbone == "resnet34":
            base = resnet34(
                weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            )
            self.out_channels = [64, 64, 128, 256, 512]

        else:
            raise ValueError(f"Unsupported backbone: '{backbone}'. Choose 'resnet50' or 'resnet34'.")

        # Optical stem: 3 → 64 (pretrained weights)
        self.optical_stem = nn.Sequential(
            base.conv1,      # Conv2d(3, 64, 7, stride=2, padding=3)
            base.bn1,
            base.relu,
            base.maxpool
        )
        
        # self.optical_stem =  nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # )
        # SAR stem: 1 → 64 (new conv — trained from scratch)
        self.sar_stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Shared encoder body
        self.layer1 = base.layer1   # ResNet-50: 256ch | ResNet-34:  64ch
        self.layer2 = base.layer2   # ResNet-50: 512ch | ResNet-34: 128ch
        self.layer3 = base.layer3   # ResNet-50:1024ch | ResNet-34: 256ch
        self.layer4 = base.layer4   # ResNet-50:2048ch | ResNet-34: 512ch

        # NOTE: self.out_channels already set above per backbone — do NOT override here

    def forward(self, x: torch.Tensor, modality: str = "optical") -> list[torch.Tensor]:
        """Returns feature maps at 5 scales: [s1, s2, s3, s4, s5]"""
        stem = self.optical_stem if modality == "optical" else self.sar_stem

        s1 = stem(x)           # (B,  64, H/4,  W/4)
        s2 = self.layer1(s1)   # (B,  C1, H/4,  W/4)
        s3 = self.layer2(s2)   # (B,  C2, H/8,  W/8)
        s4 = self.layer3(s3)   # (B,  C3, H/16, W/16)
        s5 = self.layer4(s4)   # (B,  C4, H/32, W/32)

        return [s1, s2, s3, s4, s5]