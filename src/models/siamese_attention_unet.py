"""
Siamese Attention U-Net with a pretrained ResNet-34 backbone for xView2
building-damage segmentation.

This is a *separate* model from `attention_unet.py` (which stays untouched and
usable). Key differences:

  * Bi-temporal input  : consumes BOTH pre- and post-disaster images (the
    change signal that defines damage), not post only.
  * Siamese encoder    : a single ImageNet-pretrained ResNet-34 (shared weights)
    encodes pre and post separately; features are fused per scale.
  * Decoder            : reuses the existing, already-stabilized
    `AttentionDecoderBlock` (attention gate + CBAM) from `attention_unet.py`.

Input/Output contract (drop-in for the existing `Trainer`)
----------------------------------------------------------
    forward(x) where x is (B, 6, H, W)  ==  [pre(3ch) | post(3ch)] on dim=1
    returns logits (B, num_classes, H, W)   — a plain tensor, like AttentionUNet.

So the dataset stacks pre+post into a 6-channel `image`, and `Trainer` calls
`model(image)` unchanged.

Exploding-gradient safety (the existing model had this issue; these choices keep
it controlled — see the note at the bottom of the file)
-------------------------------------------------------------------------------
  * Pretrained ResNet-34 is BN-regularized and starts in a good basin — far more
    stable than the from-scratch OpticalEncoder.
  * EVERY fusion / decoder conv is followed by BatchNorm → activations bounded.
  * Reuses the already-fixed AttentionDecoderBlock; SelfAttention (the module you
    disabled for exploding gradients) is NOT used here.
  * The change-fusion uses |pre - post|, whose subgradient is bounded (±1) and is
    immediately BN-normalized.
  * Pretrained encoder weights are NOT re-initialized (only the new
    fusion/decoder/head are Kaiming-initialised).
  * Optional `freeze_encoder_bn` to freeze BN running stats when batch size is
    small.
  * The existing Trainer safeguards (grad-clip max_norm=1.0, NaN/Inf skip,
    GradScaler self-correction) remain fully in force — nothing here bypasses them.
"""

from __future__ import annotations
import torch
import torch.nn as nn

from .attention_unet import AttentionDecoderBlock


# ImageNet statistics — pretrained ResNet expects inputs normalised this way.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class SiameseFusion(nn.Module):
    """
    Fuse pre/post features at one scale into a single feature map.

    Concatenates [pre, post, |pre - post|] and projects back to `ch` channels.
    The absolute difference injects the explicit change signal (key for damage),
    while the following BatchNorm bounds the activation magnitude → stable grads.
    """

    def __init__(self, ch: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(ch * 3, ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(pre - post)
        return self.fuse(torch.cat([pre, post, diff], dim=1))


class SiameseAttentionUNet(nn.Module):
    """
    Siamese (shared-weight) ResNet-34 encoder + change fusion + attention U-Net
    decoder.

    Args:
        num_classes:        4 (Background, Intact, Damaged, Destroyed)
        backbone:           timm backbone name (default 'resnet34')
        pretrained:         load ImageNet weights for the encoder
        dropout_p:          decoder dropout
        freeze_encoder_bn:  freeze encoder BN running stats (helps with small
                            batch sizes / extra stability)
    """

    def __init__(
        self,
        num_classes: int = 4,
        backbone: str = "resnet34",
        pretrained: bool = True,
        dropout_p: float = 0.1,
        freeze_encoder_bn: bool = False,
    ):
        super().__init__()
        import timm  # local import so the repo loads even without timm

        # Shared encoder — ResNet-34 features at strides 2/4/8/16/32.
        self.encoder = timm.create_model(
            backbone,
            features_only=True,
            pretrained=pretrained,
            out_indices=(0, 1, 2, 3, 4),
            in_chans=3,
        )
        ch = list(self.encoder.feature_info.channels())  # resnet34: [64,64,128,256,512]
        self.feature_channels = ch

        # Per-scale change fusion.
        self.fusions = nn.ModuleList([SiameseFusion(c) for c in ch])

        # Attention U-Net decoder (reuses the stabilized block).
        # f4(s32) -> dec4 -> s16, f3(s16) skip ; ... ; dec1 -> s2 ; final -> s1
        self.dec4 = AttentionDecoderBlock(in_ch=ch[4], skip_ch=ch[3], out_ch=ch[3], dropout_p=dropout_p)
        self.dec3 = AttentionDecoderBlock(in_ch=ch[3], skip_ch=ch[2], out_ch=ch[2], dropout_p=dropout_p)
        self.dec2 = AttentionDecoderBlock(in_ch=ch[2], skip_ch=ch[1], out_ch=ch[1], dropout_p=dropout_p)
        self.dec1 = AttentionDecoderBlock(in_ch=ch[1], skip_ch=ch[0], out_ch=ch[0], dropout_p=dropout_p)

        # Stride 2 -> full resolution.
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(ch[0], ch[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Conv2d(ch[0], 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1),
        )

        # Normalisation buffers (applied inside forward to each 3-ch image).
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        # Init ONLY the new modules — never overwrite pretrained encoder weights.
        self._init_decoder_weights()

        if freeze_encoder_bn:
            self._freeze_encoder_bn()

        print("[SiameseAttentionUNet] Initialized:")
        print(f"  ✓ Shared {backbone} encoder (pretrained={pretrained}) — channels {ch}")
        print("  ✓ Bi-temporal input: 6ch = [pre | post]")
        print("  ✓ Change fusion: [pre, post, |pre-post|] → 1x1 conv → BN → ReLU")
        print("  ✓ Reuses stabilized AttentionDecoderBlock (no SelfAttention)")
        print(f"  {'✓ Encoder BN frozen' if freeze_encoder_bn else '✗ Encoder BN trainable'}")

    # ------------------------------------------------------------------
    def _init_decoder_weights(self) -> None:
        """Kaiming-init the new (non-encoder) modules only."""
        new_modules = [
            self.fusions, self.dec4, self.dec3, self.dec2, self.dec1,
            self.final_upsample, self.head,
        ]
        for parent in new_modules:
            for m in parent.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _freeze_encoder_bn(self) -> None:
        for m in self.encoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    # ------------------------------------------------------------------
    def _encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = (x - self.mean) / self.std
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 6, H, W) — channels 0:3 = pre-disaster, 3:6 = post-disaster
        Returns:
            logits: (B, num_classes, H, W)
        """
        pre, post = x[:, :3], x[:, 3:]

        fp = self._encode(pre)    # 5 maps, strides 2/4/8/16/32
        fq = self._encode(post)

        # Fuse pre/post at each scale.
        f = [fuse(p, q) for fuse, p, q in zip(self.fusions, fp, fq)]

        d = self.dec4(f[4], f[3])   # s32 -> s16
        d = self.dec3(d,    f[2])   # s16 -> s8
        d = self.dec2(d,    f[1])   # s8  -> s4
        d = self.dec1(d,    f[0])   # s4  -> s2
        d = self.final_upsample(d)  # s2  -> s1
        return self.head(d)

    # ------------------------------------------------------------------
    def freeze_encoder(self, freeze: bool = True) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = not freeze


def build_siamese_attention_unet(
    num_classes: int = 4,
    backbone: str = "resnet34",
    pretrained: bool = True,
    dropout_p: float = 0.1,
    freeze_encoder_bn: bool = False,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> SiameseAttentionUNet:
    """Build a Siamese Attention U-Net and move it to `device`."""
    model = SiameseAttentionUNet(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        dropout_p=dropout_p,
        freeze_encoder_bn=freeze_encoder_bn,
    )
    return model.to(device)
