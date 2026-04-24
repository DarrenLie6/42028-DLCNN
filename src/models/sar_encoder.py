from __future__ import annotations
import torch
import torch.nn as nn

from .optical_encoder import DoubleConv 


class ResidualBlock(nn.Module):
    """
    Two Conv→BN layers with an identity skip connection.
    Helps SAR encoder learn speckle-robust features.
    in_ch == out_ch always (no projection needed).
    """
    def __init__(self, ch: int, dropout_p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.net(x))   # identity skip f(x) + x


class SAREncoderStage(nn.Module):
    """DoubleConv followed by a ResidualBlock — one encoder stage."""
    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.1):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch, dropout_p=dropout_p)
        self.res  = ResidualBlock(out_ch, dropout_p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.conv(x))


class SAREncoder(nn.Module):
    """
    Residual encoder for SAR (single-channel backscatter) imagery.
    Each stage: DoubleConv → ResidualBlock → speckle-robust features.

    out_channels = [64, 128, 256, 512]  ← same interface as OpticalEncoder
    """
    def __init__(self, dropout_p: float = 0.1):
        super().__init__()
        self.out_channels = [64, 128, 256, 512]

        self.stage1 = SAREncoderStage(1, 64,  dropout_p=dropout_p)
        self.stage2 = SAREncoderStage(64, 128, dropout_p=dropout_p)
        self.stage3 = SAREncoderStage(128, 256, dropout_p=dropout_p)
        self.stage4 = SAREncoderStage(256, 512, dropout_p=dropout_p)
        self.pool   = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Returns [s1, s2, s3, s4]:
            s1: (B,  64, H/2,  W/2)
            s2: (B, 128, H/4,  W/4)
            s3: (B, 256, H/8,  W/8)
            s4: (B, 512, H/16, W/16)
        """
        s1 = self.stage1(self.pool(x))
        s2 = self.stage2(self.pool(s1))
        s3 = self.stage3(self.pool(s2))
        s4 = self.stage4(self.pool(s3))
        return [s1, s2, s3, s4]