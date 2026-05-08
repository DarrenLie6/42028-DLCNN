# src/data/xbd_dataset.py
from __future__ import annotations
import json
import numpy as np
import cv2
import rasterio
import torch
from pathlib import Path
from torch.utils.data import Dataset
from shapely.geometry import shape
from shapely.wkt import loads as wkt_loads

from .normalization_utils import _to_float32


# Damage classification mapping (5 classes total)
XVIEW_DAMAGE_MAP = {
    "un-classified": 0,              # ← Un-classified (background)
    "no-damage":     1,              # ← No damage
    "minor-damage":  2,              # ← Minor damage
    "major-damage":  3,              # ← Major damage
    "destroyed":     4,              # ← Destroyed
}

#                bg   no-dmg  minor  major  destroyed

LABEL_NAMES = {
    0: "Un-classified",
    1: "No-Damage",
    2: "Minor-Damage",
    3: "Major-Damage",
    4: "Destroyed",
}

# xBD folder split mapping
SPLIT_DIRS = {
    "train": ["tier1", "tier3"],   # ← training uses both tiers
    "val":   ["hold"],             # ← validation
    "test":  ["test"],             # ← test
}


class XViewDataset(Dataset):
    """
    xView2/xBD dataset loader for ResNet-backed Siamese architecture.
    
    Loads pre-disaster and post-disaster optical imagery for change detection.
    No SAR data required - both images are optical (RGB).

    Folder structure expected:
        root/
          tier1/
            images/   *_pre_disaster.png  *_post_disaster.png
            labels/   *_pre_disaster.json *_post_disaster.json
          tier3/   (same structure)
          hold/    (same structure)
          test/    (same structure)
    
    Returns:
        pre_disaster: (3, H, W) - pre-disaster optical image
        post_disaster: (3, H, W) - post-disaster optical image
        label: (H, W) - damage labels with 5 classes (0-4)
        stem: image identifier
    """

    def __init__(self, root_dir: str, cfg, 
                 mode: str = "train", transform=None):
        self.root      = Path(root_dir)
        self.transform = transform
        self.mode      = mode
        self.tile_size = cfg.data.tile_size   # 256

        # ── Gather all stems from the correct split folders ───────────
        self.stems = []
        for folder in SPLIT_DIRS[mode]:
            img_dir = self.root / folder / "images"
            if not img_dir.exists():
                print(f"[XBDDataset] Warning: {img_dir} not found — skipping")
                continue
            stems = sorted([
                p.stem.replace("_pre_disaster", "")
                for p in img_dir.glob("*_pre_disaster.tif")
            ])
            # Store as (folder, stem) tuples so we know which tier
            self.stems.extend([(folder, s) for s in stems])

        print(f"[XBDDataset] mode={mode} | {len(self.stems)} tiles "
              f"from {SPLIT_DIRS[mode]}")

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        folder, stem = self.stems[idx]

        img_dir = self.root / folder / "images"
        lbl_dir = self.root / folder / "labels"

        # ── Load pre-disaster image (optical, RGB) ────────────────────
        pre_path = self._find_image(img_dir, f"{stem}_pre_disaster")
        pre = self._load_image(pre_path)

        # ── Load post-disaster image (optical, RGB) ───────────────────
        post_path = self._find_image(img_dir, f"{stem}_post_disaster")
        post = self._load_image(post_path)

        # ── Load + rasterise label (from post-disaster) ───────────────
        h, w  = post.shape[:2]
        label = self._rasterise_label(
            lbl_dir / f"{stem}_post_disaster.json", h, w
        )

        # ── Resize to standard tile size ──────────────────────────────
        if pre.shape[0] != self.tile_size or pre.shape[1] != self.tile_size:
            pre = cv2.resize(pre, (self.tile_size, self.tile_size), interpolation=cv2.INTER_LINEAR)
        
        if post.shape[0] != self.tile_size or post.shape[1] != self.tile_size:
            post = cv2.resize(post, (self.tile_size, self.tile_size), interpolation=cv2.INTER_LINEAR)
        
        if label.shape[0] != self.tile_size or label.shape[1] != self.tile_size:
            label = cv2.resize(label, (self.tile_size, self.tile_size), interpolation=cv2.INTER_NEAREST)

        # ── Augmentation ──────────────────────────────────────────────
        if self.transform:
            # Augment both pre and post images with the same transformation
            r_pre = self.transform(image=pre, mask=label)
            pre = r_pre["image"]
            label = r_pre["mask"]
            
            r_post = self.transform(image=post)
            post = r_post["image"]

        # ── To tensors ────────────────────────────────────────────────
        pre_t = torch.from_numpy(pre.transpose(2, 0, 1)).float()
        post_t = torch.from_numpy(post.transpose(2, 0, 1)).float()

        return {
            "pre_disaster":  pre_t,      # (3, H, W) - pre-disaster optical image
            "post_disaster": post_t,     # (3, H, W) - post-disaster optical image
            "label": torch.from_numpy(label).long(),
            "stem":  stem,
        }

    # ── Helpers ───────────────────────────────────────────────────────
    
    def _find_image(self, folder: Path, stem: str) -> Path:
        """Try .tif first, then .png."""
        for ext in [".tif", ".tiff", ".png"]:
            p = folder / f"{stem}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"No image found for {folder / stem}.*")


    def _load_image(self, path: Path) -> np.ndarray:
        """Load GeoTIFF or PNG → (H,W,3) float32 in [0,1]."""
        path_str = str(path)
        
        # Try rasterio for GeoTIFF files
        if path_str.endswith(('.tif', '.tiff')):
            try:
                with rasterio.open(path_str) as src:
                    img = src.read()                          # (C, H, W)
                    img = img[:3].transpose(1, 2, 0)         # (H, W, 3) — first 3 bands
                return _to_float32(img)                       # normalise to [0,1]
            except Exception as e:
                print(f"[WARNING] rasterio failed for {path}: {e}. Falling back to cv2.")
        
        # Fallback to cv2 for PNG or if rasterio fails
        img = cv2.imread(path_str, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.0

    def _rasterise_label(self, json_path: Path,
                          h: int, w: int) -> np.ndarray:
        """Parse xBD JSON polygons → (H,W) int32 damage mask with 5 classes."""
        # Use int32 to support all 5 damage classes
        mask = np.zeros((h, w), dtype=np.int32)

        if not json_path.exists():
            return mask

        with open(json_path) as f:
            data = json.load(f)

        features = data.get("features", {}).get("xy", [])   # ← pixel coords

        for feat in features:
            props      = feat.get("properties", {})
            subtype    = props.get("subtype", "no-damage")
            cls        = XVIEW_DAMAGE_MAP.get(subtype, 0)

            # xBD stores pixel-space polygons as WKT in "wkt" field
            wkt = feat.get("wkt", "")
            if not wkt:
                continue

            try:
                geom   = wkt_loads(wkt)
                coords = np.array(geom.exterior.coords, dtype=np.int32)
                cv2.fillPoly(mask, [coords], color=cls)
            except Exception:
                continue

        return mask