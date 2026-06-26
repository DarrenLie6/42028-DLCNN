"""
Transformer-based DeepLabV3+ for xView2 building-damage semantic segmentation.

Architecture -

Encoder : MiT (Mix Vision Transformer, the SegFormer backbone) — a hierarchical
          transformer with efficient (spatially-reduced) self-attention and a
          convolutional Mix-FFN. Produces 4 feature maps at strides 4/8/16/32.
Decoder : The classic DeepLabV3+ decoder — ASPP on the deepest (stride-32)
          features for multi-scale context, fused with low-level (stride-4)
          features for sharp boundaries.
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# funtional utils
def _trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    """In-place truncated-normal init (truncate at ±2 std)."""
    return nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2 * std, b=2 * std)


class DropPath(nn.Module):
    """Stochastic depth — drops the residual branch per-sample during training."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        # shape (B, 1, 1, ...) so the same mask scales every element of a sample
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x.div(keep) * mask


class DWConv(nn.Module):
    """Depth-wise 3x3 conv used inside Mix-FFN to inject local/positional info."""

    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class MixFFN(nn.Module):
    """SegFormer feed-forward: Linear -> DWConv -> GELU -> Linear."""

    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = DWConv(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class EfficientAttention(nn.Module):
    """
    Spatial-reduction multi-head self-attention (SegFormer / PVT).

    Keys and values are downsampled by ``sr_ratio`` before attention, reducing
    the O(N^2) cost to O(N * N/sr^2) — what makes transformer attention
    affordable on high-resolution segmentation feature maps.
    """

    def __init__(self, dim: int, num_heads: int, sr_ratio: int = 1, drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(drop)
        self.proj_drop = nn.Dropout(drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape

        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_)
        else:
            kv = self.kv(x)

        kv = kv.reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: Attention + Mix-FFN with stochastic depth."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        sr_ratio: int = 1,
        drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, num_heads, sr_ratio=sr_ratio, drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, int(dim * mlp_ratio), drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    """Overlapping patch embedding (strided conv) — downsamples and projects."""

    def __init__(self, patch_size: int, stride: int, in_ch: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(
            in_ch, embed_dim,
            kernel_size=patch_size, stride=stride, padding=patch_size // 2,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)                  # (B, C, H, W)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        x = self.norm(x)
        return x, H, W



# MiT (Mix Vision Transformer) encoder
class MiTEncoder(nn.Module):
    """
    Hierarchical transformer encoder producing 4 multi-scale feature maps at
    strides 4, 8, 16, 32 — the SegFormer backbone (MiT-B0 .. B2 configs).
    """

    def __init__(
        self,
        in_ch: int = 3,
        embed_dims: Tuple[int, ...] = (64, 128, 320, 512),
        num_heads: Tuple[int, ...] = (1, 2, 5, 8),
        mlp_ratios: Tuple[float, ...] = (4, 4, 4, 4),
        depths: Tuple[int, ...] = (2, 2, 2, 2),
        sr_ratios: Tuple[int, ...] = (8, 4, 2, 1),
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.depths = depths
        # Channel count of each of the 4 output stages (strides 4/8/16/32).
        # Exposed so the decoder can be built generically for any encoder.
        self.feature_channels = list(embed_dims)

        # Patch embeddings: first stage stride 4, then stride 2 each stage.
        self.patch_embeds = nn.ModuleList([
            OverlapPatchEmbed(7, 4, in_ch, embed_dims[0]),
            OverlapPatchEmbed(3, 2, embed_dims[0], embed_dims[1]),
            OverlapPatchEmbed(3, 2, embed_dims[1], embed_dims[2]),
            OverlapPatchEmbed(3, 2, embed_dims[2], embed_dims[3]),
        ])

        # Stochastic-depth decay rule — linearly increasing across all blocks.
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        self.stages = nn.ModuleList()
        self.norms = nn.ModuleList()
        cur = 0
        for i in range(4):
            blocks = nn.ModuleList([
                TransformerBlock(
                    dim=embed_dims[i],
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratios[i],
                    sr_ratio=sr_ratios[i],
                    drop=drop_rate,
                    drop_path=dpr[cur + j],
                )
                for j in range(depths[i])
            ])
            self.stages.append(blocks)
            self.norms.append(nn.LayerNorm(embed_dims[i]))
            cur += depths[i]

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            _trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            nn.init.normal_(m.weight, std=math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        B = x.shape[0]
        feats: List[torch.Tensor] = []
        for i in range(4):
            x, H, W = self.patch_embeds[i](x)
            for blk in self.stages[i]:
                x = blk(x, H, W)
            x = self.norms[i](x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
            feats.append(x)
        return feats  # strides 4, 8, 16, 32


# DeepLabV3+ decoder (ASPP + low-level fusion)
class ASPPConv(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, dilation: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Module):
    """Image-level (global) context branch of ASPP."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        x = self.conv(self.gap(x))
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling — multi-scale context aggregation."""

    def __init__(self, in_ch: int, out_ch: int = 256, rates: Tuple[int, ...] = (6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList()
        # 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ))
        # Atrous branches
        for r in rates:
            self.branches.append(ASPPConv(in_ch, out_ch, r))
        # Image pooling branch
        self.branches.append(ASPPPooling(in_ch, out_ch))

        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(rates) + 2), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = [branch(x) for branch in self.branches]
        return self.project(torch.cat(res, dim=1))


class DeepLabV3PlusDecoder(nn.Module):
    """DeepLabV3+ decoder: ASPP context + low-level skip → sharp prediction."""

    def __init__(
        self,
        high_ch: int,
        low_ch: int,
        num_classes: int,
        aspp_ch: int = 256,
        low_proj_ch: int = 48,
        aspp_rates: Tuple[int, ...] = (6, 12, 18),
    ):
        super().__init__()
        self.aspp = ASPP(high_ch, aspp_ch, rates=aspp_rates)

        # Project low-level features to a small channel count (DeepLabV3+ uses 48)
        self.low_proj = nn.Sequential(
            nn.Conv2d(low_ch, low_proj_ch, 1, bias=False),
            nn.BatchNorm2d(low_proj_ch),
            nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(aspp_ch + low_proj_ch, aspp_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(aspp_ch, aspp_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_ch),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(aspp_ch, num_classes, 1)

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        x = self.aspp(high)
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        low = self.low_proj(low)
        x = torch.cat([x, low], dim=1)
        x = self.fuse(x)
        return self.classifier(x)


class FCNAuxHead(nn.Sequential):
    """Lightweight auxiliary head (matches DeepLabV3 aux contract)."""

    def __init__(self, in_ch: int, num_classes: int):
        inter = in_ch // 4
        super().__init__(
            nn.Conv2d(in_ch, inter, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter, num_classes, 1),
        )


# Bi-temporal change fusion
class SiameseFusion(nn.Module):
    """
    Fuse pre/post encoder features at one scale into a single feature map.
    When bitemporal is True this concatanates the both features from the encoder.
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



# Pretrained encoder via timm 
# backbone varients
DEFAULT_TIMM_BACKBONE = {
    "b0": "pvt_v2_b0",
    "b1": "pvt_v2_b1",
    "b2": "pvt_v2_b2",
}


class TimmBackboneEncoder(nn.Module):
    """
    The backbone must expose 4 stages at strides 4/8/16/32 (true for the PVTv2 /
    MiT family). Override ``name`` to use any compatible timm backbone, e.g.
    'pvt_v2_b2', 'mit_b1' (if present in your timm version), 'swin_*', etc.
    """

    def __init__(self, name: str, pretrained: bool = True, in_ch: int = 3):
        super().__init__()
        import timm  # local import: module still loads if timm is absent
        self.name = name
        self.backbone = timm.create_model(
            name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            in_chans=in_ch,
        )
        self.feature_channels = list(self.backbone.feature_info.channels())
        self.reductions = list(self.backbone.feature_info.reduction())

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return list(self.backbone(x))  # 4 maps at strides 4/8/16/32


# Standard MiT (SegFormer) variant configs.
MIT_CONFIGS = {
    "b0": dict(embed_dims=(32, 64, 160, 256), depths=(2, 2, 2, 2)),
    "b1": dict(embed_dims=(64, 128, 320, 512), depths=(2, 2, 2, 2)),
    "b2": dict(embed_dims=(64, 128, 320, 512), depths=(3, 4, 6, 3)),
}


class TransformerDeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ with a Mix-Vision-Transformer (SegFormer) encoder.
    """

    def __init__(
        self,
        num_classes: int = 4,
        variant: str = "b1",
        in_ch: int = 3,
        aspp_ch: int = 256,
        aspp_rates: Tuple[int, ...] = (6, 12, 18),
        drop_path_rate: float = 0.1,
        aux_loss: bool = True,
        pretrained: bool = False,
        timm_backbone: str | None = None,
        bitemporal: bool = False,
    ):
        super().__init__()
        if variant not in MIT_CONFIGS:
            raise ValueError(f"variant must be one of {list(MIT_CONFIGS)}, got {variant!r}")

        self.num_classes = num_classes
        self.variant = variant
        self.bitemporal = bitemporal

        # Each Siamese stream still consumes a 3-channel image, so the encoder
        # is built with in_ch=3 and ImageNet-pretrained weights load cleanly.
        encoder_in_ch = 3 if bitemporal else in_ch

        # Build the encoder. With pretrained=True we use an ImageNet-pretrained
        # timm backbone; if timm is unavailable we transparently fall back to train from scratch
        # (strides 4/8/16/32), used to size the decoder for whichever encoder.
        self.encoder, feat_ch, self.encoder_kind = self._build_encoder(
            variant=variant,
            in_ch=encoder_in_ch,
            drop_path_rate=drop_path_rate,
            pretrained=pretrained,
            timm_backbone=timm_backbone,
        )

        # Per-scale change fusion (bi-temporal only). One fusion block per
        # encoder stage; each preserves that stage's channel count.
        self.fusions = (
            nn.ModuleList([SiameseFusion(c) for c in feat_ch]) if bitemporal else None
        )

        # DeepLabV3+: ASPP on deepest features (stride 32, feat_ch[3]),
        # low-level skip from the first stage (stride 4, feat_ch[0]).
        self.decoder = DeepLabV3PlusDecoder(
            high_ch=feat_ch[3],
            low_ch=feat_ch[0],
            num_classes=num_classes,
            aspp_ch=aspp_ch,
            aspp_rates=aspp_rates,
        )

        # Auxiliary head on the stride-16 features (feat_ch[2]).
        self.aux_classifier = FCNAuxHead(feat_ch[2], num_classes) if aux_loss else None

    @staticmethod
    def _build_encoder(
        variant: str,
        in_ch: int,
        drop_path_rate: float,
        pretrained: bool,
        timm_backbone: str | None,
    ):
        """Return (encoder_module, feature_channels, kind)."""
        if pretrained:
            name = timm_backbone or DEFAULT_TIMM_BACKBONE[variant]
            try:
                enc = TimmBackboneEncoder(name, pretrained=True, in_ch=in_ch)
                print(
                    f"[encoder] Using ImageNet-pretrained timm backbone '{name}' "
                    f"(channels={enc.feature_channels}, strides={enc.reductions})"
                )
                return enc, enc.feature_channels, "timm"
            except ImportError:
                print(
                    "[encoder] WARNING: `timm` is not installed - falling back to a "
                    "from-scratch MiT encoder (random init). Run `pip install timm` "
                    "to enable pretrained weights."
                )
            except Exception as e:  # bad backbone name / version mismatch
                print(
                    f"[encoder] WARNING: could not build timm backbone '{name}' ({e}). "
                    "Falling back to from-scratch MiT encoder."
                )

        cfg = MIT_CONFIGS[variant]
        enc = MiTEncoder(
            in_ch=in_ch,
            embed_dims=cfg["embed_dims"],
            depths=cfg["depths"],
            drop_path_rate=drop_path_rate,
        )
        return enc, enc.feature_channels, "scratch"

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_size = x.shape[-2:]

        if self.bitemporal:
            # x: (B, 6, H, W) = [pre(3) | post(3)]. Shared encoder over both,
            # then per-scale change fusion.
            if x.shape[1] != 6:
                raise ValueError(
                    f"bitemporal=True expects a 6-channel [pre|post] input, "
                    f"got {x.shape[1]} channels."
                )
            pre, post = x[:, :3], x[:, 3:]
            fp = self.encoder(pre)             # 4 maps, strides 4/8/16/32
            fq = self.encoder(post)            # shared weights (Siamese)
            feats = [fuse(p, q) for fuse, p, q in zip(self.fusions, fp, fq)]
        else:
            feats = self.encoder(x)            # 4 maps, strides 4/8/16/32

        c1, c2, c3, c4 = feats

        out = self.decoder(low=c1, high=c4)
        out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)

        result: Dict[str, torch.Tensor] = {"out": out}

        if self.aux_classifier is not None:
            aux = self.aux_classifier(c3)
            aux = F.interpolate(aux, size=input_size, mode="bilinear", align_corners=False)
            result["aux"] = aux

        return result

    # -- convenience API mirroring the baseline DeepLabV3Model --------------
    def set_inference_mode(self, mode: bool = True) -> None:
        self.eval() if mode else self.train()

    def freeze_backbone(self, freeze: bool = True) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = not freeze


def build_transformer_semantic_model(
    num_classes: int = 4,
    variant: str = "b1",
    aux_loss: bool = True,
    pretrained: bool = True,
    timm_backbone: str | None = None,
    bitemporal: bool = False,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> TransformerDeepLabV3Plus:
    """
    Build a transformer-based DeepLabV3+ model.

    Args:
        num_classes:   4 for xView2 (Background, Intact, Damaged, Destroyed)
        variant:       MiT encoder size — 'b0' (light), 'b1' (default), 'b2' (large)
        aux_loss:      Add an auxiliary deep-supervision head (used by the trainer)
        pretrained:    If True, load an ImageNet-pretrained timm backbone
                       (recommended). Falls back to a from-scratch MiT encoder
                       if `timm` is not installed.
        timm_backbone: Explicit timm model name.
                       Defaults to the PVTv2 model matching `variant`.
        bitemporal:    If True, expect a 6-channel [pre|post] input and fuse
                       pre/post features per scale (Siamese change detection).
        device:        Device to place the model on

    Returns:
        TransformerDeepLabV3Plus on `device`
    """
    model = TransformerDeepLabV3Plus(
        num_classes=num_classes,
        variant=variant,
        aux_loss=aux_loss,
        pretrained=pretrained,
        timm_backbone=timm_backbone,
        bitemporal=bitemporal,
    ).to(device)
    mode = "bi-temporal [pre|post]" if bitemporal else "single-input [post]"
    print(f"[model] TransformerDeepLabV3Plus mode: {mode}")
    return model
