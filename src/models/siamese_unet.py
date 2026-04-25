from __future__ import annotations
import torch
import torch.nn as nn

# from .resnet_encoder import ResNetEncoder
from .optical_encoder import OpticalEncoder, DoubleConv
from .sar_encoder import SAREncoder
from .decoder import DecoderBlock
from .projection import ProjectionFusion

"""Fully symmetric Siamese UNet:
      - OpticalEncoder: DoubleConv + SE Block per stage
      - SAREncoder:     DoubleConv + Residual Block per stage
      - ProjectionFusion at each scale to equalise + fuse skip connections
      - Standard UNet decoder consuming fused skip connections
      - CUDA stream parallelism for both encoder branchesp
"""

class SiameseUNet(nn.Module):
    
    def __init__(self, num_classes:int = 4, dropout_p: float = 0.1):
        super().__init__()
        
        # encoder 
        self.optical_encoder = OpticalEncoder(dropout_p=dropout_p)
        self.sar_encoder = SAREncoder(dropout_p=dropout_p)
        # self.encoder = ResNetEncoder(pretrained=pretrained, backbone=backbone)
        
        assert self.optical_encoder.out_channels == self.sar_encoder.out_channels, \
                f"Encoder channel mismatch: " \
                f"optical={self.optical_encoder.out_channels} " \
                f"sar={self.sar_encoder.out_channels}"
        
        # each skip with pre features post feature to double the channels
        enc_ch = self.optical_encoder.out_channels
        proj_ch = [ch // 2 for ch in enc_ch]
        
        # projection fusion
        self.proj4 = ProjectionFusion(enc_ch[3], enc_ch[3], proj_ch[3])
        self.proj3 = ProjectionFusion(enc_ch[2], enc_ch[2], proj_ch[2])
        self.proj2 = ProjectionFusion(enc_ch[1], enc_ch[1], proj_ch[1])
        self.proj1 = ProjectionFusion(enc_ch[0], enc_ch[0], proj_ch[0])
        
        # bottleneck
        bottleneck_in = proj_ch[3] * 2  # derived: 256*2=512 or 128*2=256
        self.bottleneck = nn.Sequential(
            DoubleConv(bottleneck_in, bottleneck_in // 2),
            nn.Dropout2d(p=0.4)
        )
        bottleneck_out = bottleneck_in // 2     # also derived

        # decoder
        self.dec3 = DecoderBlock(bottleneck_out, proj_ch[2] * 2, proj_ch[2], dropout_p=dropout_p)
        self.dec2 = DecoderBlock(proj_ch[2],     proj_ch[1] * 2, proj_ch[1], dropout_p=dropout_p)
        self.dec1 = DecoderBlock(proj_ch[1],     proj_ch[0] * 2, proj_ch[0], dropout_p=dropout_p)

        # Final upsample H/2 → H (no skip, just upsample + conv)
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(proj_ch[0], proj_ch[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(proj_ch[0]),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Conv2d(proj_ch[0], 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1)
        )
        
    def forward(self,optical: torch.Tensor, sar: torch.Tensor, optical_valid: torch.Tensor) -> torch.Tensor:
        

        # parallel encoder branches (parallel process the 2 encoder branches)
        # if optical.is_cuda:
        #     stream_opt = torch.cuda.Stream()
        #     stream_sar = torch.cuda.Stream()

        #     with torch.cuda.stream(stream_opt):
        #         feats_opt = self.optical_encoder(optical)

        #     with torch.cuda.stream(stream_sar):
        #         feats_sar = self.sar_encoder(sar)

        #     torch.cuda.synchronize()
        # else:
        feats_opt = self.optical_encoder(optical)
        feats_sar = self.sar_encoder(sar)
        
        # optical features when optical is missing 
        mask = optical_valid.float().view(-1, 1, 1, 1)  # (B,1,1,1)
        feats_opt = [f * mask for f in feats_opt]

        # projection fusion at each scale
        fused4 = self.proj4(feats_opt[3], feats_sar[3])  # (B, 512, H/16, W/16)
        fused3 = self.proj3(feats_opt[2], feats_sar[2])  # (B, 256, H/8,  W/8)
        fused2 = self.proj2(feats_opt[1], feats_sar[1])  # (B, 128, H/4,  W/4)
        fused1 = self.proj1(feats_opt[0], feats_sar[0])  # (B,  64, H/2,  W/2)

        # bottleneck
        x = self.bottleneck(fused4)          # (B, 256, H/16, W/16)
        
        # decoder
        x = self.dec3(x, fused3) # (B, 128, H/8,  W/8)
        x = self.dec2(x, fused2) # (B,  64, H/4,  W/4)
        x = self.dec1(x, fused1) # (B,  32, H/2,  W/2)

        x = self.final_upsample(x) # (B,  32, H,    W)

        return self.head(x) # (B,  num_classes, H, W)