from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from .normalization_utils import load_optical, load_sar
from .augmentation_utils import build_train_aug, build_val_aug


class BRIGHTDataset(Dataset):

    def __init__(self, root_dir, split_file, cfg, transform=None, mode="train"):
        self.root = Path(root_dir)
        self.cfg = cfg
        self.transform = transform
        self.mode = mode

        self.opt_mean = np.array(cfg.data.optical_mean, dtype=np.float32)
        self.opt_std = np.array(cfg.data.optical_std, dtype=np.float32)
        self.sar_mean = np.array(cfg.data.sar_mean, dtype=np.float32)
        self.sar_std = np.array(cfg.data.sar_std, dtype=np.float32)

        with open(split_file) as f:
            self.stems = [l.strip() for l in f if l.strip()]
        if not self.stems:
            raise ValueError(f"Split file {split_file} is empty.")

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, index):
        stem = self.stems[index]
        opt_path = self.root / self.cfg.data.pre_event_dir  / f"{stem}_pre_disaster.tif"
        sar_path = self.root / self.cfg.data.post_event_dir / f"{stem}_post_disaster.tif"
        lbl_path = self.root / self.cfg.data.target_dir     / f"{stem}_building_damage.tif"

        tile_size = self.cfg.data.tile_size

        # Optical 
        optical_valid = opt_path.exists()        # False for ukraine/myanmar/mexico
        optical = load_optical(opt_path, tile_size=tile_size)  # zeros if missing

        # SAR 
        sar = load_sar(sar_path, tile_size=tile_size)

        # Label 
        if lbl_path.exists():
            label = rasterio.open(lbl_path).read(1).astype(np.int64)
            if label.shape[0] != tile_size or label.shape[1] != tile_size:
                label = cv2.resize(
                    label, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST
                )
        else:
            label = np.zeros((tile_size, tile_size), dtype=np.int64)

        # Augmentations 
        if self.transform:
            r = self.transform(image=optical, sar=sar, mask=label)
            optical = r["image"]
            sar = r["sar"]
            label = r["mask"]

        # Normalisation
        optical = (optical - self.opt_mean) / self.opt_std
        sar = (sar - self.sar_mean) / self.sar_std

        return {
            "optical":       torch.from_numpy(optical.transpose(2, 0, 1)).float(),
            "optical_valid": torch.tensor(optical_valid, dtype=torch.bool),
            "sar":           torch.from_numpy(sar.transpose(2, 0, 1)).float(),
            "label":         torch.from_numpy(label).long(),
            "stem":          stem,
        }