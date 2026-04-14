from __future__ import annotations
import os
from pathlib import Path

import numpy as np
import rasterio
import cv2

# Normalization helper functions
def _to_float32(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    return arr.astype(np.float32)

def load_optical(path: str | Path, tile_size: int = 512) -> np.ndarray:
    """Read pre event optical GeoTIFF (HxWx3 float32 in [0,1])"""
    path = Path(path)
    
    # Return dummy data if file doesn't exist (for smoke tests)
    if not path.exists():
        print(f"Warning: optical file not found {path}, returning dummy data")
        return np.zeros((tile_size, tile_size, 3), dtype=np.float32)
    
    with rasterio.open(path) as src:
        img = src.read() #(C, H, W)
        img = np.transpose(img[:3], (1, 2, 0)) #(H, W, 3)
    
    img = _to_float32(img)
    
    # Resize to standard tile size
    if img.shape[0] != tile_size or img.shape[1] != tile_size:
        img = cv2.resize(img, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
    
    return img

def load_sar(path: str | Path, tile_size: int = 512) -> np.ndarray:
    """Read post-event SAR GeoTIFF (HxWx1 float32 in [0,1])"""
    path = Path(path)
    
    # Return dummy data if file doesn't exist (for smoke tests)
    if not path.exists():
        print(f"Warning: SAR file not found {path}, returning dummy data")
        return np.zeros((tile_size, tile_size, 1), dtype=np.float32)
    
    with rasterio.open(path) as src:
        img = src.read(1).astype(np.float32) #(H,W)
    img = np.log1p(img) #log-compress heavy tail
    p2, p98 = np.percentile(img, 2), np.percentile(img, 98)
    img = np.clip(img, p2, p98)
    img = (img - p2) / ((p98 - p2) + 1e-8)
    
    # Resize to standard tile size
    if img.shape[0] != tile_size or img.shape[1] != tile_size:
        img = cv2.resize(img, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
    
    return img[:, :, np.newaxis] #(H, W, 1)

