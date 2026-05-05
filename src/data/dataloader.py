from __future__ import annotations
from pathlib import Path
from collections import defaultdict

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import BRIGHTDataset
from .augmentation_utils import build_train_aug, build_val_aug


# ── Split file builder (BRIGHT only) ─────────────────────────────────────────
def build_pairs(cfg) -> None:
    """
    Scans the train/pre-event folder and writes train_set.txt, val_set.txt
    if the split files do not exist. BRIGHT only.
    """
    splits_dir = Path(cfg.data.split_file_dir)
    train_txt  = splits_dir / cfg.data.train_split
    val_txt    = splits_dir / cfg.data.val_split

    if train_txt.exists() and val_txt.exists():
        return

    splits_dir.mkdir(parents=True, exist_ok=True)

    pre_event_dir = Path(cfg.data.root_dir) / cfg.data.pre_event_dir
    stems = sorted([
        p.stem.replace("_pre_disaster", "")
        for p in pre_event_dir.glob("*_pre_disaster.tif")
    ])

    if not stems:
        raise FileNotFoundError(f"No tiles found in {pre_event_dir}")

    events = defaultdict(list)
    for stem in stems:
        event = "_".join(stem.split("_")[:-1])
        events[event].append(stem)

    train_stems, val_stems = [], []
    for event_stems in events.values():
        n_val = max(1, int(len(event_stems) * 0.2))
        val_stems.extend(event_stems[:n_val])
        train_stems.extend(event_stems[n_val:])

    train_txt.write_text("\n".join(train_stems))
    val_txt.write_text("\n".join(val_stems))
    print(f"[build_pairs] wrote {len(train_stems)} train / {len(val_stems)} val stems")


# ── Weighted Sampler (BRIGHT only) ────────────────────────────────────────────
def _compute_tile_weights_bright(dataset: BRIGHTDataset, cfg) -> torch.Tensor:
    """
    Assigns each tile a sampling weight proportional to its damaged pixel
    fraction. Tiles with destroyed damage get a 2x boost.
    """
    weights        = []
    damage_classes = {2, 3}

    for stem in dataset.stems:
        lbl_path = (
            dataset.root / cfg.data.target_dir / f"{stem}_building_damage.tif"
        )

        if not lbl_path.exists():
            weights.append(1.0)
            continue

        with rasterio.open(lbl_path) as src:
            label = src.read(1)

        total   = label.size
        damaged = np.isin(label, list(damage_classes)).sum()
        frac    = damaged / total
        severe  = (label == 3).sum() / total
        weight  = frac + 2.0 * severe + 1e-6

        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)


# ── Weighted Sampler (xBD) ────────────────────────────────────────────────────
def _compute_tile_weights_xbd(dataset, cfg) -> torch.Tensor:
    """
    Assigns each xBD tile a sampling weight based on damage pixel fraction.
    Labels are rasterised on-the-fly from JSON — use cached stems.
    """
    weights        = []
    damage_classes = {2, 3}

    for folder, stem in dataset.stems:
        lbl_path = (
            Path(cfg.data.root_dir) / folder / "labels"
            / f"{stem}_post_disaster.json"
        )

        if not lbl_path.exists():
            weights.append(1.0)
            continue

        # Load rasterised label using dataset helper
        h = w = cfg.data.tile_size
        label = dataset._rasterise_label(lbl_path, h, w)

        total   = label.size
        damaged = np.isin(label, list(damage_classes)).sum()
        frac    = damaged / total
        severe  = (label == 3).sum() / total
        weight  = frac + 2.0 * severe + 1e-6

        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)


# ── Collate ───────────────────────────────────────────────────────────────────
def collate_fn(batch):
    return {
        "optical":       torch.stack([b["optical"]       for b in batch]),
        "optical_valid": torch.stack([b["optical_valid"] for b in batch]),
        "sar":           torch.stack([b["sar"]           for b in batch]),
        "label":         torch.stack([b["label"]         for b in batch]),
        "stem":          [b["stem"] for b in batch],
    }


# ── Main entry point ──────────────────────────────────────────────────────────
def get_dataloaders(cfg):
    """
    Returns (train_loader, val_loader, test_loader).
    Supports cfg.data.dataset = 'bright' | 'xbd'
    """
    dataset_name = getattr(cfg.data, "dataset", "bright").lower()

    # ── BRIGHT ────────────────────────────────────────────────────────
    if dataset_name == "bright":
        build_pairs(cfg)
        splits_dir = Path(cfg.data.split_file_dir)

        train_ds = BRIGHTDataset(
            root_dir   = cfg.data.root_dir,
            split_file = splits_dir / cfg.data.train_split,
            cfg        = cfg,
            transform  = build_train_aug(cfg),
            mode       = "train",
        )

        tile_weights = _compute_tile_weights_bright(train_ds, cfg)
        sampler      = WeightedRandomSampler(
            weights     = tile_weights,
            num_samples = len(tile_weights),
            replacement = True,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size         = cfg.training.batch_size,
            sampler            = sampler,
            num_workers        = cfg.training.num_workers,
            pin_memory         = True,
            persistent_workers = True,
            prefetch_factor    = 2,
            drop_last          = True,
        )

        val_ds = BRIGHTDataset(
            root_dir   = cfg.data.root_dir,
            split_file = splits_dir / cfg.data.val_split,
            cfg        = cfg,
            transform  = build_val_aug(cfg),
            mode       = "val",
        )
        val_loader = DataLoader(
            val_ds,
            batch_size         = cfg.training.batch_size,
            shuffle            = False,
            num_workers        = cfg.training.num_workers,
            pin_memory         = True,
            persistent_workers = True,
            prefetch_factor    = 2,
        )

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

        print(f"[BRIGHT] train={len(train_ds)} val={len(val_ds)} tiles")

    # ── xBD ───────────────────────────────────────────────────────────
    elif dataset_name == "xview":
        from src.data.xview2_dataset import XViewDataset

        train_ds = XViewDataset(
            root_dir  = cfg.data.root_dir,
            cfg       = cfg,
            mode      = "train",
            transform = build_train_aug(cfg),
        )

        tile_weights = _compute_tile_weights_xbd(train_ds, cfg)
        sampler      = WeightedRandomSampler(
            weights     = tile_weights,
            num_samples = len(tile_weights),
            replacement = True,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size         = cfg.training.batch_size,
            sampler            = sampler,
            num_workers        = cfg.training.num_workers,
            pin_memory         = True,
            persistent_workers = True,
            prefetch_factor    = 2,
            drop_last          = True,
        )

        val_ds = XViewDataset(
            root_dir  = cfg.data.root_dir,
            cfg       = cfg,
            mode      = "val",
            transform = build_val_aug(cfg),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size         = cfg.training.batch_size,
            shuffle            = False,
            num_workers        = cfg.training.num_workers,
            pin_memory         = True,
            persistent_workers = True,
            prefetch_factor    = 2,
        )

        test_ds = XViewDataset(
            root_dir  = cfg.data.root_dir,
            cfg       = cfg,
            mode      = "test",
            transform = build_val_aug(cfg),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size  = cfg.training.batch_size,
            shuffle     = False,
            num_workers = cfg.training.num_workers,
            pin_memory  = True,
        )

        print(f"[xBD] train={len(train_ds)} val={len(val_ds)} "
              f"test={len(test_ds)} tiles")

    else:
        raise ValueError(f"Unknown dataset: '{dataset_name}'. "
                         f"Choose 'bright' or 'xbd'.")

    return train_loader, val_loader, test_loader