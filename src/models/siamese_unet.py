from __future__ import annotations
import torch
import torch.nn as nn

from .encoder import ResNetEncoder
from .decoder import DecoderBlock, DoubleConv

"""Siamese UNet - Optical-Optical Architecture
    - dual encoder branches with ResNet50 weights  
    - both branches process optical (RGB) imagery
    - features fused by concatanation at each scale
    - UNet decoder outputs 4 class segmentation map (damage levels)
"""

class SiameseUNet(nn.Module):
    
    def __init__(self, num_classes:int = 4, pretrained: bool = True):
        super().__init__()
        self.encoder = ResNetEncoder(pretrained=pretrained)
        
        # each skip with pre features post feature to double the channels
        enc_ch = self.encoder.out_channels # [64, 256, 512, 1024, 2048]
        
        # bottleneck fusion: concat pre + post > 2048*2
        self.bottleneck_conv = nn.Sequential(
            DoubleConv(enc_ch[4] * 2, 1024),
            nn.Dropout2d(p=0.3)
            )
        
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
        self, 
        pre_disaster: torch.Tensor,    # (B, 3, H, W) - pre-disaster optical
        post_disaster: torch.Tensor    # (B, 3, H, W) - post-disaster optical
    ) -> torch.Tensor:
        """
        Args:
            pre_disaster: (B, 3, H, W) - pre-disaster optical image
            post_disaster: (B, 3, H, W) - post-disaster optical image
            
        Returns:
            (B, num_classes, H, W) - damage segmentation map
        """
        
        # encode both optical branches through shared encoder
        feats_pre = self.encoder(pre_disaster)
        feats_post = self.encoder(post_disaster)
        
        # fuse the bottleneck
        bottleneck = torch.cat([feats_pre[4], feats_post[4]], dim=1) # 2048 * 2
        x = self.bottleneck_conv(bottleneck)
        
        # decode with skip connections
        # each skip = concat(pre, post)
        skip4 = torch.cat([feats_pre[3], feats_post[3]], dim=1)
        skip3 = torch.cat([feats_pre[2], feats_post[2]], dim=1)
        skip2 = torch.cat([feats_pre[1], feats_post[1]], dim=1)
        skip1 = torch.cat([feats_pre[0], feats_post[0]], dim=1)
        
        x = self.dec4(x, skip4) #512ch, H/16
        x = self.dec3(x, skip3) #256ch, H/8
        x = self.dec2(x, skip2) #128ch, H/4
        x = self.dec1(x, skip1) #64ch, H/2
        x = self.final_upsample(x) #64ch, H
                
        return self.head(x)