from __future__ import annotations
import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

class ResNetEncoder(nn.Module):
    """ResNet34 encoder - Shared ResNet34 encoder for optical imagery.
    
    Both pre-disaster and post-disaster images are optical (RGB).
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = resnet34(
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        )
        
        # Optical stem: 3 channels (RGB) → 64 channels
        self.optical_stem = nn.Sequential(
            backbone.conv1, #Conv2D(3, 64, 7, stride=2, padding=3)
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        
        # shared encoder body
        self.layer1 = backbone.layer1 # 64, stride 4
        self.layer2 = backbone.layer2 # 128, stride 8
        self.layer3 = backbone.layer3 # 256, stride 16
        self.layer4 = backbone.layer4 # 512, stride 32
        
        # output channels at each scale (used for decoding) - ResNet34 channels
        self.out_channels = [64, 64, 128, 256, 512]
        
    def forward(self, x: torch.Tensor):
        """Returns a list of feature maps [s1, s2, s3, s4, s5]
        
        Args:
            x: (B, 3, H, W) - optical image (RGB)
            
        Returns:
            List of feature maps at each scale
        """
        s1 = self.optical_stem(x) #(B, 64, H/4, W/4)
        s2 = self.layer1(s1) #(B, 64, H/4, W/4)
        s3 = self.layer2(s2) #(B, 128, H/8, W/8)
        s4 = self.layer3(s3) #(B, 256, H/16, W/16)
        s5 = self.layer4(s4) #(B, 512, H/32, W/32)
        
        return [s1, s2, s3, s4, s5]