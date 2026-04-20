from __future__ import annotations
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNetEncoder(nn.Module):
    """Resnet50 encoder - Shared ResNet50 encoder with seperate input stems
    """
    
    def __intit__(self, pretrained: bool = True):
        super().__init__()
        backbone = resnet50(
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )
        
        # seperate STEMS
        # optical stem: 3 > 64
        self.optical_stem = nn.Sequential(
            backbone.conv1, #Conv2D(3, 64, 7, stride=2, padding=3)
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        # SAR stem: 1 > 64 (new conv- not pretrained)
        self.sar_stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # shared encoder body
        self.layer1 = backbone.layer1 # 256, stride 4
        self.layer2 = backbone.layer2 # 512, stride 8
        self.layer3 = backbone.layer3 # 1024, stride 16
        self.layer4 = backbone.layer4 # 2048, stride 32
        
        # output channels at each scale (used for decoding)
        self.out_channels = [64, 256, 512, 1024, 2048]
        
    def forward(self, x: torch.Tensor, modality: str = "optical"):
        """Returns a list of feature maps [s1, s2, s3, s4, s5]
        """
        if modality == "optical":
            stem = self.optical_stem
        else:
            stem = self.sar_stem
        
        s1 = stem(x) #(B, 64, H/2, W/2) - before maxpooling 
        s2 = self.layer1(s1) #(B, 256, H/4, W/4)
        s3 = self.layer1(s2) #(B, 512, H/8, W/8)
        s4 = self.layer1(s3) #(B, 1024, H/16, W/16)
        s5 = self.layer1(s5) #(B, 2048, H/32, W/32)
        
        return [s1, s2, s3, s4, s5]