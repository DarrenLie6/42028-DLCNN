from __future__ import annotations
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import BRIGHTDataset
from .augmentation_utils import build_train_aug, build_val_aug

# Split file builder
def build_pairs(cfg) -> None:
    """
    Scans the train/pre-event folder and writes train_set.txt, val_set.txt
    if the split files is not exists
    """
    
    splits_dir = Path(cfg.data.split_file_dir)
    train_txt  = splits_dir / cfg.data.train_split
    val_txt    = splits_dir / cfg.data.val_split

    if train_txt.exists() and val_txt.exists():
        return   # official splits already present return None

    splits_dir.mkdir(parents=True, exist_ok=True)

    # Gather all stems from train pre-event folder
    pre_event_dir = Path(cfg.data.root_dir) / cfg.data.pre_event_dir
    stems = sorted([
        p.stem.replace("_pre_disaster", "")
        for p in pre_event_dir.glob("*_pre_disaster.tif")
    ])

    if not stems:
        raise FileNotFoundError(f"No tiles found in {pre_event_dir}")

    # Group stems by event name (everything before the last underscore+id)
    from collections import defaultdict
    events = defaultdict(list)
    for stem in stems:
        event = "_".join(stem.split("_")[:-1])
        events[event].append(stem)

    # Stratified 80/20 split per event
    train_stems, val_stems = [], []
    for event_stems in events.values():
        n_val = max(1, int(len(event_stems) * 0.2))
        val_stems.extend(event_stems[:n_val])
        train_stems.extend(event_stems[n_val:])

    train_txt.write_text("\n".join(train_stems))
    val_txt.write_text("\n".join(val_stems))
    print(f"[build_pairs] wrote {len(train_stems)} train / {len(val_stems)} val stems")
    
# Weighted Sampler
def _compute_tile_weights(dataset: BRIGHTDataset, cfg) -> torch.Tensor:
    """
    Assigns each tile a sampling weight proportional to its damaged pixel fraction. 
    Tiles with major/destroyed damage (4 classes) get a 2x boost.
    Replaces shuffle=True to combat severe class imbalance.
    """
    weights = []
    damage_classes = {2, 3}  # exclude background

    for stem in dataset.stems:
        lbl_path = (dataset.root / cfg.data.target_dir / f"{stem}_building_damage.tif")
        
        # Use uniform weight if label file doesn't exist (for smoke tests)
        if not lbl_path.exists():
            weights.append(1.0)
            continue
            
        with rasterio.open(lbl_path) as src:
            label = src.read(1)

        total   = label.size
        damaged = np.isin(label, list(damage_classes)).sum()
        frac    = damaged / total

        # 2x boost for tiles with severe damage (class 4)
        severe  = (label == 3).sum() / total 
        weight  = frac + 2.0 * severe + 1e-6   # +1e-6 avoids zero weights

        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)

def collate_fn(batch):
    return {
        "optical": torch.stack([b["optical"] for b in batch]),
        "optical_valid": torch.stack([b["optical_valid"] for b in batch]),
        "sar": torch.stack([b["sar"]   for b in batch]),
        "label": torch.stack([b["label"]   for b in batch]),
        "stem": [b["stem"]   for b in batch],
    }

# Data Loader
def get_dataloaders(cfg):
    """Returns train/val/test loaders"""
    
    # build split files if not present
    build_pairs(cfg)
    
    splits_dir = Path(cfg.data.split_file_dir)
    
    # Train dataset + weighted sampler
    train_ds = BRIGHTDataset(
        root_dir=cfg.data.root_dir,
        split_file= splits_dir / cfg.data.train_split,
        cfg= cfg,
        transform= build_train_aug(cfg),
        mode = "train",
    )
    
    # compute train set weight
    tiles_weights = _compute_tile_weights(train_ds, cfg)
    # pasing the tiles weights to WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights= tiles_weights,
        num_samples= len(tiles_weights),
        replacement= True
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size= cfg.training.batch_size,
        sampler= sampler,
        num_workers= cfg.training.num_workers,
        pin_memory= True,
        persistent_workers= True,
        prefetch_factor= 2,
        drop_last= True
    )
    
    # val dataset
    val_ds = BRIGHTDataset(
        root_dir   = cfg.data.root_dir,
        split_file = splits_dir / cfg.data.val_split,
        cfg        = cfg,
        transform  = build_val_aug(cfg),
        mode       = "val",
    )

    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg.training.batch_size,
        shuffle     = False,
        num_workers = cfg.training.num_workers,
        pin_memory  = True,
        persistent_workers= True,
        prefetch_factor= 2,
    )
    
    # test dataset
    test_split = splits_dir / cfg.data.test_split
    if test_split.exists():
        test_ds = BRIGHTDataset(
            root_dir   = cfg.data.root_dir,
            split_file = test_split,
            cfg        = cfg,
            transform  = build_val_aug(cfg),
            mode       = "test",
        )
        test_loader = DataLoader(
            test_ds,
            batch_size  = cfg.training.batch_size,
            shuffle     = False,
            num_workers = cfg.training.num_workers,
            pin_memory  = True,
        )
    else:
        test_loader = None
        print("[get_dataloaders] No test split found — test_loader is None")
    
    print(f"[get_dataloaders] train={len(train_ds)} val={len(val_ds)} tiles")
    return train_loader, val_loader, test_loader