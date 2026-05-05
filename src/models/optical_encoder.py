from __future__ import annotations
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
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

    def forward(self, x):
        return self.net(x)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.
    Globally pools spatial dims → FC → sigmoid → reweights channels.
    reduction=8 keeps it lightweight.
    """
    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(ch // reduction, ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)        # (B, C) 2d matrix
        w = self.fc(w).view(b, c, 1, 1)   # (B, C, 1, 1)
        return x * w                        # channel-wise reweighting


class OpticalEncoderStage(nn.Module):
    """DoubleConv followed by an SE block — one encoder stage."""
    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.1):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch, dropout_p=dropout_p)
        self.se   = SEBlock(out_ch, reduction=8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.conv(x))


class OpticalEncoder(nn.Module):
    """
    Mid-level encoder for optical (RGB) imagery.
    Each stage: DoubleConv → SE Block → channel recalibration.

    out_channels = [64, 128, 256, 512]
    """
    def __init__(self, dropout_p: float = 0.1):
        super().__init__()
        self.out_channels = [64, 128, 256, 512]

        self.stage1 = OpticalEncoderStage(3, 64,  dropout_p=dropout_p)
        self.stage2 = OpticalEncoderStage(64, 128, dropout_p=dropout_p)
        self.stage3 = OpticalEncoderStage(128, 256, dropout_p=dropout_p)
        self.stage4 = OpticalEncoderStage(256, 512, dropout_p=dropout_p)
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