from __future__ import annotations
import torch
import torch.nn as nn

from .optical_encoder import OpticalEncoder, DoubleConv
from .decoder import DecoderBlock


class UNet(nn.Module):
    """
    Standard UNet with single encoder:
    - Takes only post-disaster image (3-channel RGB)
    - Single OpticalEncoder for feature extraction
    - Standard decoder with skip connections
    - Outputs semantic segmentation map
    """

    def __init__(self, num_classes: int = 4, dropout_p: float = 0.1):
        super().__init__()

        # ── Encoder ───────────────────────────────────────────────────
        self.encoder = OpticalEncoder(dropout_p=dropout_p)
        enc_ch = self.encoder.out_channels  # [64, 128, 256, 512]

        # ── Bottleneck ────────────────────────────────────────────────
        bottleneck_in = enc_ch[3]   # 512
        bottleneck_out = bottleneck_in // 2  # 256

        self.bottleneck = nn.Sequential(
            DoubleConv(bottleneck_in, bottleneck_out),
            nn.Dropout2d(p=0.5),  # Increased from 0.4 to improve regularization
        )

        # ── Decoder ───────────────────────────────────────────────────
        self.dec3 = DecoderBlock(bottleneck_out, enc_ch[2], enc_ch[2], dropout_p=dropout_p)
        self.dec2 = DecoderBlock(enc_ch[2], enc_ch[1], enc_ch[1], dropout_p=dropout_p)
        self.dec1 = DecoderBlock(enc_ch[1], enc_ch[0], enc_ch[0], dropout_p=dropout_p)

        # ── Final upsample H/2 → H ────────────────────────────────────
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(enc_ch[0], enc_ch[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(enc_ch[0]),
            nn.ReLU(inplace=True),
        )

        # ── Segmentation Head ─────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Conv2d(enc_ch[0], 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1),
        )

        print("[UNet] Initialized — single encoder + standard decoder")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, 3, H, W) — post-disaster RGB image

        Returns:
            logits: (B, num_classes, H, W) — segmentation logits
        """

        # ── Encoder forward ───────────────────────────────────────────
        feats = self.encoder(image)  # List of 4 feature maps

        # ── Bottleneck ────────────────────────────────────────────────
        x = self.bottleneck(feats[3])  # (B, 256, H/16, W/16)

        # ── Decoder with skip connections ─────────────────────────────
        x = self.dec3(x, feats[2])  # (B, 256, H/8, W/8)
        x = self.dec2(x, feats[1])  # (B, 128, H/4, W/4)
        x = self.dec1(x, feats[0])  # (B, 64, H/2, W/2)

        x = self.final_upsample(x)  # (B, 64, H, W)

        return self.head(x)  # (B, num_classes, H, W)
