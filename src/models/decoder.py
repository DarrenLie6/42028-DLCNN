from __future__ import annotations
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Two consecutive Conv → BN → ReLU blocks.
    Defined here independently so decoder.py has no circular imports.
    """
    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    """
    One UNet decoder step:
        1. Upsample x by 2x (bilinear)
        2. Concatenate with skip connection from ProjectionFusion
        3. DoubleConv to reduce channels
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout_p: float = 0.1):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2,
            mode='bilinear',
            align_corners=False
        )
        self.conv = DoubleConv(
            in_ch  = in_ch + skip_ch,   # concat along channel dim
            out_ch = out_ch,
            dropout_p = dropout_p
        )

    def forward(self,x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)             # (B, in_ch, H*2, W*2)
        x = torch.cat([x, skip], dim=1)  # (B, in_ch + skip_ch, H*2, W*2)
        
        return self.conv(x)              # (B, out_ch,   H*2, W*2)