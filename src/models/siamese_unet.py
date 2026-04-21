from __future__ import annotations
import torch
import torch.nn as nn

from .encoder import ResNetEncoder
from .decoder import DecoderBlock, DoubleConv

"""Siamese UNet
    - dual encoder branches with ResNet50 weights
    - featurs fused by concatanation at each scale
    - UNet decoder outputs 5 class segmentation map
"""

class SiameseUNet(nn.Module):
    
    def __init__(self, num_classes:int = 5, pretrained: bool = True):
        super().__init__()
        self.encoder = ResNetEncoder(pretrained=pretrained)
        
        # each skip with pre features post feature to double the channels
        enc_ch = self.encoder.out_channels # [64, 256, 512, 1024, 2048]
        
        # bottleneck fusion: conacat pre + post > 2048*2
        self.bottleneck_conv = DoubleConv(enc_ch[4] *  2, 1024)
        
        # decoder blocks 
        # skp_ch  = pre channels + post channels 
        self.dec4 = DecoderBlock(1024, enc_ch[3] * 2, 512)
        self.dec3 = DecoderBlock(512, enc_ch[2] * 2, 256)
        self.dec2 = DecoderBlock(256, enc_ch[1] * 2, 128)
        self.dec1 = DecoderBlock(128, enc_ch[0] * 2, 64)
        
        # final upsample x2 back to original resolutioin + classification head
        self.final_upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False
        )
        
        self.head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
        
    def forward(
        self, optical: torch.Tensor, # (B, 3, H, W)
        sar: torch.Tensor, # (B, 1, H, W)
        optical_valid: torch.Tensor
    ) -> torch.Tensor:
        
        # encodes the both branches
        feats_opt = self.encoder(optical, modality="optical")
        feats_sar = self.encoder(sar, modality="sar")
        
        # gate optical features where modality is missing
        # optical_valid: (B,) > (B, 1, 1, 1) for broadcasting
        mask = optical_valid.float().view(-1, 1, 1, 1)
        feats_opt = [f * mask for f in feats_opt]
        
        # fuse the bottleneck
        bottleneck = torch.cat([feats_opt[4], feats_sar[4]], dim=1) #2048 * 2
        x = self.bottleneck_conv(bottleneck)
        
        # decode with skip connections
        # each skip = concat(prescale, post scale)
        skip4 = torch.cat([feats_opt[3], feats_sar[3]], dim=1)
        skip3 = torch.cat([feats_opt[2], feats_sar[2]], dim=1)
        skip2 = torch.cat([feats_opt[1], feats_sar[1]], dim=1)
        skip1 = torch.cat([feats_opt[0], feats_sar[0]], dim=1)
        
        # x = self.dec4(x, skip4) #512ch, H/16
        # x = self.dec3(x, skip3) #256ch, H/8
        # x = self.dec2(x, skip2) #128ch, H/4
        # x = self.dec1(x, skip1) #64ch, H/2
        # x = self.final_upsample(x) #64ch, H
        
        print("bottleneck:", x.shape)
        x = self.dec4(x, skip4); print("dec4:", x.shape)
        x = self.dec3(x, skip3); print("dec3:", x.shape)
        x = self.dec2(x, skip2); print("dec2:", x.shape)
        x = self.dec1(x, skip1); print("dec1:", x.shape)
        x = self.final_upsample(x); print("final_upsample:", x.shape)
                
        return self.head(x)