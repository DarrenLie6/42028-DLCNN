# from __future__ import annotations
# import torch
# import torch.nn as nn

# from .optical_encoder import OpticalEncoder, DoubleConv
# from .decoder import DecoderBlock
# from .attention import AttentionGate, CBAM, SelfAttention


# class AttentionDecoderBlock(nn.Module):
#     """
#     Enhanced decoder block with attention gate on skip connection.
    
#     1. Upsample x
#     2. Apply attention gate to skip connection
#     3. Concatenate
#     4. DoubleConv + CBAM for refinement
#     """
#     def __init__(
#         self,
#         in_ch: int,
#         skip_ch: int,
#         out_ch: int,
#         dropout_p: float = 0.1,
#         use_attention_gate: bool = True
#     ):
#         super().__init__()
#         self.use_attention_gate = use_attention_gate
        
#         # self.upsample = nn.Upsample(
#         #     scale_factor=2,
#         #     mode='bilinear',
#         #     align_corners=False
#         # )
        
#         if self.use_attention_gate:
#             # Attention gate for skip connection
#             # f_g = in_ch (gating signal), f_l = skip_ch (skip signal)
#             self.attention_gate = AttentionGate(
#                 f_g=in_ch,
#                 f_l=skip_ch,
#                 f_int=max(skip_ch // 2, 16)
#             )
        
#         self.conv = DoubleConv(
#             in_ch=in_ch + skip_ch,
#             out_ch=out_ch,
#             dropout_p=dropout_p
#         )
        
#         self.cbam = CBAM(out_ch, reduction=8, kernel_size=7)

#     def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
#         x = self.upsample(x)
        
#         if self.use_attention_gate:
#             skip = self.attention_gate(x, skip)
        
#         x = torch.cat([x, skip], dim=1)
#         x = self.conv(x)
#         x = self.cbam(x)
#         return x


# class AttentionUNet(nn.Module):
#     """
#     Attention UNet with multi-scale attention mechanisms (numerically stable version).
    
#     Features:
#     - Attention gates on all skip connections (suppress irrelevant features)
#     - CBAM (Channel + Spatial Attention) in decoder blocks
#     - Optional self-attention at bottleneck (disabled by default for stability)
#     - CBAM in bottleneck for feature refinement
#     - Improved skip connection quality through gating
    
#     This reduces loss by:
#     1. Focusing on relevant spatial regions (spatial attention)
#     2. Emphasizing important channels (channel attention)
#     3. Suppressing false positive activations (attention gates)
#     4. Capturing distant context (self-attention if enabled)
#     """

#     def __init__(
#         self,
#         num_classes: int = 4,
#         dropout_p: float = 0.1,
#         use_self_attention: bool = False  # ← Disabled by default for stability
#     ):
#         super().__init__()
#         self.use_self_attention = use_self_attention

#         # ── Encoder ───────────────────────────────────────────────────
#         self.encoder = OpticalEncoder(dropout_p=dropout_p)
#         enc_ch = self.encoder.out_channels  # [64, 128, 256, 512]

#         # ── Bottleneck with Optional Self-Attention ─────────────────
#         bottleneck_in = enc_ch[3]    # 512
#         bottleneck_out = bottleneck_in // 2  # 256

#         self.bottleneck = nn.Sequential(
#             DoubleConv(bottleneck_in, bottleneck_out),
#             nn.Dropout2d(p=0.5),
#         )
        
#         # Add self-attention at bottleneck only if explicitly enabled
#         if self.use_self_attention:
#             self.bottleneck_attention = SelfAttention(
#                 in_channels=bottleneck_out,
#                 inter_channels=max(bottleneck_out // 16, 1)  # Even more reduced
#             )
        
#         # Add CBAM for additional refinement
#         self.bottleneck_cbam = CBAM(bottleneck_out, reduction=4)

#         # ── Decoder with Attention Gates ──────────────────────────────
#         self.dec3 = AttentionDecoderBlock(
#             in_ch=bottleneck_out,
#             skip_ch=enc_ch[2],
#             out_ch=enc_ch[2],
#             dropout_p=dropout_p,
#             use_attention_gate=True
#         )
#         self.dec2 = AttentionDecoderBlock(
#             in_ch=enc_ch[2],
#             skip_ch=enc_ch[1],
#             out_ch=enc_ch[1],
#             dropout_p=dropout_p,
#             use_attention_gate=True
#         )
#         self.dec1 = AttentionDecoderBlock(
#             in_ch=enc_ch[1],
#             skip_ch=enc_ch[0],
#             out_ch=enc_ch[0],
#             dropout_p=dropout_p,
#             use_attention_gate=True
#         )

#         # ── Final upsample H/2 → H ────────────────────────────────────
#         self.final_upsample = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
#             nn.Conv2d(enc_ch[0], enc_ch[0], kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(enc_ch[0]),
#             nn.ReLU(inplace=True),
#         )
        
#         # Add CBAM before final head
#         self.final_cbam = CBAM(enc_ch[0], reduction=8)

#         # ── Segmentation Head ─────────────────────────────────────────
#         self.head = nn.Sequential(
#             nn.Conv2d(enc_ch[0], 16, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(16),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(16, num_classes, kernel_size=1),
#         )

#         self_attn_str = "✓ Self-attention at bottleneck" if self.use_self_attention else "✗ Self-attention disabled (stable mode)"
#         print("[AttentionUNet] Initialized with:")
#         print("  ✓ Attention gates on all skip connections")
#         print("  ✓ CBAM (Channel + Spatial) in decoder blocks")
#         print(f"  {self_attn_str}")
#         print("  ✓ Enhanced bottleneck with CBAM")
#         print("  ✓ Numerically stable (LayerNorm + clamping)")

#     def forward(self, image: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             image: (B, 3, H, W) — post-disaster RGB image

#         Returns:
#             logits: (B, num_classes, H, W) — segmentation logits
#         """

#         # ── Encoder forward ───────────────────────────────────────────
#         feats = self.encoder(image)  # List of 4 feature maps

#         # ── Bottleneck with optional attention ────────────────────────
#         x = self.bottleneck(feats[3])        # (B, 256, H/16, W/16)
#         if self.use_self_attention:
#             x = self.bottleneck_attention(x)     # Self-attention (if enabled)
#         x = self.bottleneck_cbam(x)          # Channel + spatial attention

#         # ── Decoder with attention gates on skip connections ──────────
#         x = self.dec3(x, feats[2])  # (B, 256, H/8, W/8)   + attention gate
#         x = self.dec2(x, feats[1])  # (B, 128, H/4, W/4)   + attention gate
#         x = self.dec1(x, feats[0])  # (B, 64, H/2, W/2)    + attention gate

#         x = self.final_upsample(x)  # (B, 64, H, W)
#         x = self.final_cbam(x)      # Final refinement with CBAM

#         return self.head(x)  # (B, num_classes, H, W)

from __future__ import annotations
import torch
import torch.nn as nn

from .optical_encoder import OpticalEncoder, DoubleConv
from .decoder import DecoderBlock
from .attention import AttentionGate, CBAM, SelfAttention


class AttentionDecoderBlock(nn.Module):
    """
    Enhanced decoder block with attention gate on skip connection.

    Order:
      1. Upsample x  (x goes from H/2 to H to match skip)
      2. Apply attention gate — g=upsampled x, x=skip (now same spatial size)
      3. Concatenate
      4. DoubleConv + CBAM
    """
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        dropout_p: float = 0.1,
        use_attention_gate: bool = True
    ):
        super().__init__()
        self.use_attention_gate = use_attention_gate

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        if self.use_attention_gate:
            # After upsampling, x has in_ch channels — use as gating signal
            self.attention_gate = AttentionGate(
                f_g=in_ch,
                f_l=skip_ch,
                f_int=max(skip_ch // 2, 16)
            )

        self.conv = DoubleConv(
            in_ch=in_ch + skip_ch,
            out_ch=out_ch,
            dropout_p=dropout_p
        )

        self.cbam = CBAM(out_ch, reduction=8, kernel_size=7)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # x:    (B, in_ch,   H/2, W/2)
        # skip: (B, skip_ch, H,   W)

        x = self.upsample(x)                      # (B, in_ch,   H, W)  ← upsample FIRST

        if self.use_attention_gate:
            # Both x and skip are now at the same spatial size (H, W)
            skip = self.attention_gate(x, skip)   # (B, skip_ch, H, W)

        x = torch.cat([x, skip], dim=1)           # (B, in_ch + skip_ch, H, W)
        x = self.conv(x)                           # (B, out_ch, H, W)
        x = self.cbam(x)
        return x


class AttentionUNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        dropout_p: float = 0.1,
        use_self_attention: bool = False
    ):
        super().__init__()
        self.use_self_attention = use_self_attention

        # ── Encoder ───────────────────────────────────────────────────
        self.encoder = OpticalEncoder(dropout_p=dropout_p)
        enc_ch = self.encoder.out_channels   # [64, 128, 256, 512]

        # ── Bottleneck ────────────────────────────────────────────────
        bottleneck_in  = enc_ch[3]           # 512
        bottleneck_out = bottleneck_in // 2  # 256

        self.bottleneck = nn.Sequential(
            DoubleConv(bottleneck_in, bottleneck_out),
            nn.Dropout2d(p=0.5),
        )

        if self.use_self_attention:
            self.bottleneck_attention = SelfAttention(
                in_channels=bottleneck_out,
                inter_channels=max(bottleneck_out // 16, 1)
            )

        self.bottleneck_cbam = CBAM(bottleneck_out, reduction=8)

        # ── Decoder ───────────────────────────────────────────────────
        self.dec3 = AttentionDecoderBlock(
            in_ch=bottleneck_out,   # 256, H/16
            skip_ch=enc_ch[2],      # 256, H/8
            out_ch=enc_ch[2],       # 256
            dropout_p=dropout_p
        )
        self.dec2 = AttentionDecoderBlock(
            in_ch=enc_ch[2],        # 256, H/8
            skip_ch=enc_ch[1],      # 128, H/4
            out_ch=enc_ch[1],       # 128
            dropout_p=dropout_p
        )
        self.dec1 = AttentionDecoderBlock(
            in_ch=enc_ch[1],        # 128, H/4
            skip_ch=enc_ch[0],      # 64,  H/2
            out_ch=enc_ch[0],       # 64
            dropout_p=dropout_p
        )

        # ── Final upsample H/2 → H ────────────────────────────────────
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(enc_ch[0], enc_ch[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(enc_ch[0]),
            nn.ReLU(inplace=True),
        )

        self.final_cbam = CBAM(enc_ch[0], reduction=8)

        # ── Segmentation Head ─────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Conv2d(enc_ch[0], 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1),
        )

        self._initialize_weights()

        self_attn_str = "✓ Self-attention at bottleneck" if self.use_self_attention else "✗ Self-attention disabled"
        print("[AttentionUNet] Initialized:")
        print("  ✓ Attention gates — upsample before gate (spatial fix)")
        print("  ✓ AttentionGate — BN → ReLU → Conv → Sigmoid")
        print("  ✓ ChannelAttention — min-channel guard (max(ch//r, 4))")
        print("  ✓ ReLU inplace=False after residual additions")
        print(f"  {self_attn_str}")
        print("  ✓ Kaiming weight initialisation")

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image:  (B, 3, H, W)
        Returns:
            logits: (B, num_classes, H, W)
        """
        # ── Encoder ───────────────────────────────────────────────────
        feats = self.encoder(image)
        # feats[0]: (B,  64, H/2,  W/2)
        # feats[1]: (B, 128, H/4,  W/4)
        # feats[2]: (B, 256, H/8,  W/8)
        # feats[3]: (B, 512, H/16, W/16)

        # ── Bottleneck ────────────────────────────────────────────────
        x = self.bottleneck(feats[3])         # (B, 256, H/16, W/16)
        if self.use_self_attention:
            x = self.bottleneck_attention(x)
        x = self.bottleneck_cbam(x)

        # ── Decoder ───────────────────────────────────────────────────
        x = self.dec3(x, feats[2])   # upsample→(H/8),  gate, cat, conv → (B, 256, H/8,  W/8)
        x = self.dec2(x, feats[1])   # upsample→(H/4),  gate, cat, conv → (B, 128, H/4,  W/4)
        x = self.dec1(x, feats[0])   # upsample→(H/2),  gate, cat, conv → (B,  64, H/2,  W/2)

        x = self.final_upsample(x)   # (B, 64, H, W)
        x = self.final_cbam(x)

        return self.head(x)          # (B, num_classes, H, W)