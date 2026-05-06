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


XVIEW_DAMAGE_MAP = {
    "no-damage":     1,              # ← Intact (annotated background)
    "minor-damage":  2,              # ← Damaged
    "major-damage":  2,              # ← Damaged
    "destroyed":     3,              # ← Destroyed
    "un-classified": -100,           # ← Ignore in loss/metrics
}

# xBD folder split mapping
SPLIT_DIRS = {
    "train": ["tier1", "tier3"],   # ← training uses both tiers
    "val":   ["hold"],             # ← validation
    "test":  ["test"],             # ← test
}


class XViewDataset(Dataset):
    """
    xView2/xBD dataset loader.

    Folder structure expected:
        root/
          tier1/
            images/   *_pre_disaster.png  *_post_disaster.png
            labels/   *_pre_disaster.json *_post_disaster.json
          tier3/   (same structure)
          hold/    (same structure)
          test/    (same structure)
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

        # ── Load post-disaster image (RGB only) ───────────────────────
        post_path = self._find_image(img_dir, f"{stem}_post_disaster")
        post = self._load_image(post_path)

        # ── Load + rasterise label ────────────────────────────────────
        h, w  = post.shape[:2]
        label = self._rasterise_label(
            lbl_dir / f"{stem}_post_disaster.json", h, w
        )

        # ── Augmentation ──────────────────────────────────────────────
        if self.transform:
            # For SimpleUNet, we only augment the post-disaster image
            r     = self.transform(image=post, mask=label)
            post  = r["image"]
            label = r["mask"]

        # ── To tensors ────────────────────────────────────────────────
        post_t = torch.from_numpy(post.transpose(2, 0, 1)).float()

        return {
            "image": post_t,
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
        """Parse xBD JSON polygons → (H,W) int32 damage mask with -100 for un-classified."""
        # Use int32 to support -100 for un-classified regions
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