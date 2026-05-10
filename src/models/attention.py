# from __future__ import annotations
# import torch
# import torch.nn as nn


# class ChannelAttention(nn.Module):
#     """
#     Channel Attention Module (from CBAM).
#     Uses both Average Pooling and Max Pooling to generate channel attention.
#     Helps network focus on important feature channels.
#     """
#     def __init__(self, in_channels: int, reduction: int = 8):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
        
#         self.fc = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         out = avg_out + max_out
#         return x * self.sigmoid(out)


# class SpatialAttention(nn.Module):
#     """
#     Spatial Attention Module (from CBAM).
#     Uses channel-wise statistics to generate spatial attention maps.
#     Helps network focus on important spatial regions.
#     kernel_size=7 is standard; can be tuned based on your feature map sizes.
#     """
#     def __init__(self, kernel_size: int = 7):
#         super().__init__()
#         padding = (kernel_size - 1) // 2
#         self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x_cat = torch.cat([avg_out, max_out], dim=1)
#         out = self.sigmoid(self.conv(x_cat))
#         return x * out


# class CBAM(nn.Module):
#     """
#     Convolutional Block Attention Module (CBAM).
#     Combines Channel and Spatial attention sequentially.
#     Lightweight and effective for improving feature quality.
#     """
#     def __init__(self, in_channels: int, reduction: int = 8, kernel_size: int = 7):
#         super().__init__()
#         self.ca = ChannelAttention(in_channels, reduction=reduction)
#         self.sa = SpatialAttention(kernel_size=kernel_size)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.ca(x)
#         x = self.sa(x)
#         return x


# class AttentionGate(nn.Module):
#     """
#     Attention Gate for UNet decoder (from Attention UNet paper).
#     Gates the skip connection by learning to suppress irrelevant features.
    
#     Args:
#         f_g: channels in gating signal (from lower decoder layer)
#         f_l: channels in input/skip signal (from encoder)
#         f_int: intermediate channels for attention computation
#     """
#     def __init__(self, f_g: int, f_l: int, f_int: int):
#         super().__init__()
#         self.gate = nn.Sequential(
#             nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(f_int),
#         )
#         self.skip = nn.Sequential(
#             nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(f_int),
#         )
#         self.combine = nn.Sequential(
#             nn.ReLU(inplace=True),
#             nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )
        
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

#     def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             g: gating signal (from current decoder layer)
#             x: skip signal (from encoder)
        
#         Returns:
#             gated skip connection
#         """
#         g1 = g1 = self.upsample(self.gate(g))
#         x1 = self.skip(x)
#         combined = g1 + x1
#         psi = self.combine(combined)
#         return x * psi


# class SelfAttention(nn.Module):
#     """
#     Numerically stable Self-Attention Module.
#     Uses careful scaling and log-softmax to prevent NaN.
    
#     Args:
#         in_channels: number of input channels
#         inter_channels: intermediate channels (reduced for stability)
#     """
#     def __init__(self, in_channels: int, inter_channels: int = None):
#         super().__init__()
#         if inter_channels is None:
#             inter_channels = max(in_channels // 16, 1)  # ← Even smaller for stability
        
#         self.in_channels = in_channels
#         self.inter_channels = inter_channels
#         self.scale = (inter_channels ** -0.5)  # Proper scaling factor
        
#         # Query, Key, Value projections
#         self.query_conv = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
#         self.key_conv = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
#         self.value_conv = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        
#         # Add layer norm for stability
#         self.query_norm = nn.LayerNorm(inter_channels)
#         self.key_norm = nn.LayerNorm(inter_channels)
        
#         # Output projection
#         self.out_conv = nn.Conv2d(inter_channels, in_channels, kernel_size=1, bias=False)
#         self.out_norm = nn.LayerNorm(in_channels)
        
#         # Learnable residual weight (start at 0 for stability)
#         self.gamma = nn.Parameter(torch.zeros(1))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Numerically stable attention computation.
#         Args:
#             x: (B, C, H, W)
#         Returns:
#             refined features (B, C, H, W)
#         """
#         batch_size, channels, height, width = x.size()
        
#         # Project to query, key, value
#         query = self.query_conv(x)  # (B, C', H, W)
#         key = self.key_conv(x)      # (B, C', H, W)
#         value = self.value_conv(x)  # (B, C', H, W)
        
#         # Reshape for layer norm
#         query_flat = query.permute(0, 2, 3, 1).reshape(-1, self.inter_channels)  # (B*H*W, C')
#         key_flat = key.permute(0, 2, 3, 1).reshape(-1, self.inter_channels)      # (B*H*W, C')
        
#         # Apply layer norm for numerical stability
#         query_flat = self.query_norm(query_flat)
#         key_flat = self.key_norm(key_flat)
        
#         # Reshape back
#         query = query_flat.reshape(batch_size, height, width, self.inter_channels).permute(0, 3, 1, 2)
#         key = key_flat.reshape(batch_size, height, width, self.inter_channels).permute(0, 3, 1, 2)
        
#         # Flatten spatial dimensions
#         query = query.view(batch_size, self.inter_channels, -1)  # (B, C', H*W)
#         key = key.view(batch_size, self.inter_channels, -1)      # (B, C', H*W)
#         value = value.view(batch_size, self.inter_channels, -1)  # (B, C', H*W)
        
#         # Transpose for attention computation
#         key = key.permute(0, 2, 1)  # (B, H*W, C')
        
#         # Compute attention scores with proper scaling
#         attention_scores = torch.bmm(key, query)  # (B, H*W, H*W)
#         attention_scores = attention_scores * self.scale  # Scale down to prevent overflow
        
#         # Use log_softmax then exp for numerical stability
#         attention = torch.nn.functional.softmax(
#             attention_scores.clamp(min=-50, max=50),  # Clamp to prevent inf
#             dim=-1
#         )
        
#         # Apply attention to values
#         out = torch.bmm(value, attention)  # (B, C', H*W)
#         out = out.view(batch_size, self.inter_channels, height, width)  # (B, C', H, W)
#         out = self.out_conv(out)  # (B, C, H, W)
        
#         # Apply output layer norm
#         out_norm = out.permute(0, 2, 3, 1).reshape(-1, channels)
#         out_norm = self.out_norm(out_norm)
#         out = out_norm.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        
#         # Residual connection with learnable weight (starts at 0, grows during training)
#         return x + self.gamma * out


# class AttentionResidualBlock(nn.Module):
#     """
#     Residual block with CBAM attention.
#     Combines residual learning with channel and spatial attention.
#     """
#     def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.1):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout2d(p=dropout_p)
        
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
        
#         self.cbam = CBAM(out_channels)
        
#         # Skip connection adjustment if needed
#         self.skip_conv = None
#         if in_channels != out_channels:
#             self.skip_conv = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         identity = x
        
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.dropout(out)
        
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.cbam(out)
        
#         if self.skip_conv is not None:
#             identity = self.skip_conv(identity)
        
#         out = out + identity
#         out = self.relu(out)
#         return out


from __future__ import annotations
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Guard: never reduce below 4 channels
        reduced = max(in_channels // reduction, 4)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, reduced, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, in_channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * out


class CBAM(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 8, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x


class AttentionGate(nn.Module):
    """
    Attention Gate for UNet decoder.

    Receives g and x at the SAME spatial resolution.
    The caller (AttentionDecoderBlock) is responsible for upsampling
    g to match x before calling this gate.

    Args:
        f_g:   channels in gating signal
        f_l:   channels in skip signal
        f_int: intermediate channels
    """
    def __init__(self, f_g: int, f_l: int, f_int: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(f_int),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(f_int),
        )
        # FIX: correct order is BN → ReLU → Conv → Sigmoid
        self.combine = nn.Sequential(
            nn.BatchNorm2d(f_int),
            nn.ReLU(inplace=True),
            nn.Conv2d(f_int, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # NO upsample here — caller handles spatial alignment

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: gating signal  (B, f_g, H, W) — already upsampled to match x
            x: skip signal    (B, f_l, H, W)
        Returns:
            gated skip        (B, f_l, H, W)
        """
        g1 = self.gate(g)          # (B, f_int, H, W)
        x1 = self.skip(x)          # (B, f_int, H, W)
        psi = self.combine(g1 + x1)  # (B, 1, H, W)
        return x * psi


class SelfAttention(nn.Module):
    def __init__(self, in_channels: int, inter_channels: int = None):
        super().__init__()
        if inter_channels is None:
            inter_channels = max(in_channels // 16, 1)

        self.in_channels    = in_channels
        self.inter_channels = inter_channels
        self.scale          = inter_channels ** -0.5

        self.query_conv = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.key_conv   = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.value_conv = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)

        self.query_norm = nn.LayerNorm(inter_channels)
        self.key_norm   = nn.LayerNorm(inter_channels)

        self.out_conv = nn.Conv2d(inter_channels, in_channels, kernel_size=1, bias=False)
        self.out_norm = nn.LayerNorm(in_channels)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()

        query = self.query_conv(x)
        key   = self.key_conv(x)
        value = self.value_conv(x)

        query_flat = self.query_norm(query.permute(0,2,3,1).reshape(-1, self.inter_channels))
        key_flat   = self.key_norm(key.permute(0,2,3,1).reshape(-1, self.inter_channels))

        query = query_flat.reshape(B, H, W, self.inter_channels).permute(0,3,1,2)
        key   = key_flat.reshape(B, H, W, self.inter_channels).permute(0,3,1,2)

        query = query.view(B, self.inter_channels, -1)
        key   = key.view(B, self.inter_channels, -1).permute(0, 2, 1)
        value = value.view(B, self.inter_channels, -1)

        attn = torch.nn.functional.softmax(
            (torch.bmm(key, query) * self.scale).clamp(-50, 50), dim=-1
        )

        out = self.out_conv(torch.bmm(value, attn).view(B, self.inter_channels, H, W))
        out_flat = self.out_norm(out.permute(0,2,3,1).reshape(-1, C))
        out = out_flat.reshape(B, H, W, C).permute(0,3,1,2)

        return x + self.gamma * out


class AttentionResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.1):
        super().__init__()
        self.conv1   = nn.Conv2d(in_channels,  out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1     = nn.BatchNorm2d(out_channels)
        self.relu    = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.conv2   = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2     = nn.BatchNorm2d(out_channels)
        self.cbam    = CBAM(out_channels)

        self.skip_conv = None
        if in_channels != out_channels:
            self.skip_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # FIX: inplace=False after residual add to protect identity branch gradient
        self.relu_out = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)

        if self.skip_conv is not None:
            identity = self.skip_conv(identity)

        return self.relu_out(out + identity)
