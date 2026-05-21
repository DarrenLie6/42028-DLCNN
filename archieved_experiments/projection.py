from __future__ import annotations
import torch
import torch.nn as nn


class ProjectionFusion(nn.Module):
    """
    Projects optical and SAR features to equal channels
    then concatenates them for the skip connection.

    opt_ch + sar_ch → proj_ch + proj_ch = proj_ch * 2
    """

    def __init__(self, opt_ch: int, sar_ch: int, proj_ch: int):
        super().__init__()
        self.proj_opt = nn.Sequential(
            nn.Conv2d(opt_ch, proj_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(proj_ch),
            nn.ReLU(inplace=True),
        )
        self.proj_sar = nn.Sequential(
            nn.Conv2d(sar_ch, proj_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(proj_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat_opt: torch.Tensor, feat_sar: torch.Tensor) -> torch.Tensor:
        
        return torch.cat([
            self.proj_opt(feat_opt),   # (B, proj_ch, H, W)
            self.proj_sar(feat_sar),   # (B, proj_ch, H, W)
        ], dim=1)                      # (B, proj_ch*2, H, W)