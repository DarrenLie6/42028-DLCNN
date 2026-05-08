from __future__ import annotations
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .xview2_dataset import XViewDataset
from .augmentation_utils import build_train_aug, build_val_aug

# Split file builder - NOT NEEDED for XViewDataset (uses tier1/tier3/hold/test folders)
# XViewDataset handles splitting automatically via SPLIT_DIRS mapping

    
# Weighted Sampler
def _compute_tile_weights(dataset: XViewDataset, cfg) -> torch.Tensor:
    """
    Assigns each tile a sampling weight proportional to its damaged pixel fraction. 
    Tiles with major/destroyed damage get a 2x boost.
    """
    weights = []
    damage_classes = {2, 3, 4}  # minor, major, destroyed (exclude background and no-damage)

    for folder, stem in dataset.stems:
        lbl_dir = dataset.root / folder / "labels"
        lbl_path = lbl_dir / f"{stem}_post_disaster.json"
        
        # Use uniform weight if label file doesn't exist
        if not lbl_path.exists():
            weights.append(1.0)
            continue
        
        # Parse damage from JSON and compute weight
        try:
            import json
            with open(lbl_path) as f:
                data = json.load(f)
            
            features = data.get("features", {}).get("xy", [])
            damage_pixels = 0
            for feat in features:
                props = feat.get("properties", {})
                subtype = props.get("subtype", "no-damage")
                if subtype in ["minor-damage", "major-damage", "destroyed"]:
                    damage_pixels += 1
            
            frac = damage_pixels / len(features) if features else 0
            weight = frac + 1e-6
        except:
            weight = 1.0

        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)

def collate_fn(batch):
    """Collate batch for Siamese architecture with pre/post-disaster images."""
    return {
        "pre_disaster": torch.stack([b["pre_disaster"] for b in batch]),
        "post_disaster": torch.stack([b["post_disaster"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "stem": [b["stem"] for b in batch],
    }

# Data Loader
def get_dataloaders(cfg):
    """Returns train/val/test loaders using XViewDataset"""
    
    # Train dataset + weighted sampler
    train_ds = XViewDataset(
        root_dir=cfg.data.root_dir,
        cfg=cfg,
        mode="train",
        transform=build_train_aug(cfg),
    )
    
    # compute train set weight
    tiles_weights = _compute_tile_weights(train_ds, cfg)
    # passing the tiles weights to WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=tiles_weights,
        num_samples=len(tiles_weights),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        sampler=sampler,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    
    # val dataset
    val_ds = XViewDataset(
        root_dir=cfg.data.root_dir,
        cfg=cfg,
        mode="val",
        transform=build_val_aug(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # test dataset
    test_ds = XViewDataset(
        root_dir=cfg.data.root_dir,
        cfg=cfg,
        mode="test",
        transform=build_val_aug(),
    )
    
    if len(test_ds) > 0:
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
    else:
        test_loader = None
        print("[get_dataloaders] No test samples found — test_loader is None")
    
    print(f"[get_dataloaders] train={len(train_ds)} val={len(val_ds)} tiles")
    return train_loader, val_loader, test_loader