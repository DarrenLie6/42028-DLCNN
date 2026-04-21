from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """UNet decoder
        - upsampled features from previous decoder
        - concat skip connection from both branches
        - Conv > BN > ReLU > Conv > Bn > ReLU
    """
    
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.block(x)
    
class DecoderBlock(nn.Module):
    """Upsample - concat skip connection > DoubleConv
        skp_ch: total channels from skip connections 
    """
    def __init__(self, in_ch: int, skp_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.conv = DoubleConv(in_ch + skp_ch, out_ch)
        
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # handles size mismatch from odd dimensions
        if x.shape[-2:] != skip.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
            
        x = torch.cat([x, skip], dim=1)
        
        return self.conv(x)