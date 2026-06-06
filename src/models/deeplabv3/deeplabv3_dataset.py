"""
Semantic segmentation dataset adapter for xView2.
Converts xView2 polygon annotations to per-pixel damage classification.
"""

from __future__ import annotations
import json
import numpy as np
import cv2
import rasterio
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, Optional

from src.data.normalization_utils import _to_float32


# Damage class mapping for semantic segmentation
XVIEW_DAMAGE_MAP = {
    "no-damage":     1,       # Intact
    "minor-damage":  2,       # Damaged
    "major-damage":  2,       # Damaged
    "destroyed":     3,       # Destroyed
    "un-classified": -100,    # Ignore in loss
}

# xBD folder split mapping
SPLIT_DIRS = {
    "train": ["tier1", "tier3"],
    "val":   ["hold"],
    "test":  ["test"],
}


def _parse_wkt_polygon(wkt: str) -> Optional[np.ndarray]:
    """
    Parse a WKT POLYGON string into a pixel-coordinate array for cv2.

    xView2 stores building footprints as WKT strings in pixel space, e.g.:
        "POLYGON ((x1 y1, x2 y2, x3 y3, ...))"

    Returns:
        (N, 1, 2) int32 array suitable for cv2.fillPoly, or None on failure.
    """
    try:
        # Strip WKT wrapper — handles both POLYGON and MULTIPOLYGON edge cases
        wkt = wkt.strip()
        if not wkt.upper().startswith("POLYGON"):
            return None

        # Extract the first ring: content between the outermost (( and ))
        start = wkt.index("((") + 2
        end   = wkt.index("))", start)
        coords_str = wkt[start:end]

        coords = []
        for pair in coords_str.split(","):
            parts = pair.strip().split()
            if len(parts) >= 2:
                coords.append([int(float(parts[0])), int(float(parts[1]))])

        if len(coords) < 3:
            return None

        # cv2.fillPoly expects shape (N, 1, 2)
        return np.array(coords, dtype=np.int32).reshape((-1, 1, 2))

    except Exception:
        return None


def build_semantic_mask_from_xview(
    json_path: Path,
    h: int,
    w: int,
) -> np.ndarray:
    """
    Convert xView2 polygon annotations to a semantic segmentation mask.

    xView2 JSON structure:
        {
          "features": {
            "xy": [
              {
                "properties": {"subtype": "no-damage", ...},
                "wkt": "POLYGON ((x1 y1, x2 y2, ...))"   ← pixel coords
              },
              ...
            ]
          }
        }

    Args:
        json_path: Path to xView2 JSON annotation file
        h: Image height in pixels
        w: Image width in pixels

    Returns:
        mask: (H, W) int32 array — 0=Background, 1=Intact, 2=Damaged, 3=Destroyed
    """
    mask = np.zeros((h, w), dtype=np.int32)

    if not json_path.exists():
        return mask

    try:
        with open(json_path) as f:
            data = json.load(f)

        # xView2 canonical structure: data["features"]["xy"]
        features = []
        if isinstance(data, dict):
            raw = data.get("features", {})
            if isinstance(raw, dict):
                features = raw.get("xy", [])
            elif isinstance(raw, list):
                features = raw
        elif isinstance(data, list):
            features = data

        for idx, feature in enumerate(features):
            if idx > 1000:  # Safety cap
                break

            if not isinstance(feature, dict):
                continue

            # Damage class from properties
            props = feature.get("properties", {})
            if not isinstance(props, dict):
                continue

            subtype = props.get("subtype") or props.get("damage", "un-classified")
            if subtype not in XVIEW_DAMAGE_MAP:
                subtype = "un-classified"

            label = XVIEW_DAMAGE_MAP[subtype]
            if label == -100:
                continue

            # ── WKT path (xView2 native format) ──────────────────────────
            wkt = feature.get("wkt", "")
            if wkt:
                polygon = _parse_wkt_polygon(wkt)
                if polygon is not None:
                    cv2.fillPoly(mask, [polygon], label)
                continue

            # ── GeoJSON fallback (geometry.coordinates) ───────────────────
            geom   = feature.get("geometry", {})
            coords = geom.get("coordinates", []) if isinstance(geom, dict) else []
            gtype  = geom.get("type", "") if isinstance(geom, dict) else ""

            if gtype == "Polygon" and coords:
                polygon = np.array(coords[0], dtype=np.int32).reshape((-1, 1, 2))
                if polygon.shape[0] >= 3:
                    cv2.fillPoly(mask, [polygon], label)

    except Exception:
        pass  # Return blank background mask for corrupt files

    return mask


class SemanticSegmentationXViewDataset(Dataset):
    """
    xView2 dataset loader for semantic segmentation.

    Single input: Post-disaster images
    Output: Per-pixel damage classification (Background, Intact, Damaged, Destroyed)
    """

    def __init__(
        self,
        root_dir: str,
        cfg,
        mode: str = "train",
        transform=None,
    ):
        self.root      = Path(root_dir)
        self.transform = transform
        self.mode      = mode
        self.tile_size = cfg.data.tile_size if hasattr(cfg.data, "tile_size") else 512

        self.stems = []
        skipped    = 0

        for folder in SPLIT_DIRS[mode]:
            img_dir = self.root / folder / "images"
            lbl_dir = self.root / folder / "labels"

            if not img_dir.exists():
                continue

            for p in sorted(img_dir.glob("*_post_disaster.tif")):
                stem      = p.stem.replace("_post_disaster", "")
                json_path = lbl_dir / f"{stem}_post_disaster.json"

                if json_path.exists() and not self._is_valid_json(json_path):
                    skipped += 1
                    continue

                self.stems.append((folder, stem))

        print(
            f"[SemanticSegmentationXViewDataset] mode={mode} | "
            f"{len(self.stems)} tiles from {SPLIT_DIRS[mode]}"
            + (f" | {skipped} malformed labels skipped" if skipped else "")
        )

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        folder, stem = self.stems[idx]

        img_dir = self.root / folder / "images"
        lbl_dir = self.root / folder / "labels"

        post_path = self._find_image(img_dir, f"{stem}_post_disaster")
        image     = self._load_image(post_path)
        h, w      = image.shape[:2]

        json_path = lbl_dir / f"{stem}_post_disaster.json"
        label     = build_semantic_mask_from_xview(json_path, h, w)

        # Resize to tile size
        if h != self.tile_size or w != self.tile_size:
            image = cv2.resize(
                image, (self.tile_size, self.tile_size),
                interpolation=cv2.INTER_LINEAR,
            )
            label = cv2.resize(
                label, (self.tile_size, self.tile_size),
                interpolation=cv2.INTER_NEAREST,
            )

        # Augmentation
        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image = transformed["image"]
            label = transformed["mask"]

        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float()
        label_t = torch.from_numpy(np.array(label, dtype=np.int32)).long()

        return {"image": image_t, "label": label_t, "stem": stem}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_valid_json(json_path: Path) -> bool:
        """
        Returns True if the JSON is parseable and all features are dicts.
        Accepts tiles with zero features (background-only tiles are valid).
        """
        try:
            with open(json_path) as f:
                data = json.load(f)

            if isinstance(data, dict):
                raw      = data.get("features", {})
                features = raw.get("xy", []) if isinstance(raw, dict) else raw
            elif isinstance(data, list):
                features = data
            else:
                return False

            return all(isinstance(f, dict) for f in features)

        except Exception:
            return False

    def _find_image(self, folder: Path, stem: str) -> Path:
        for ext in [".tif", ".tiff", ".png"]:
            p = folder / f"{stem}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"No image found for {folder / stem}.*")

    def _load_image(self, path: Path) -> np.ndarray:
        """Load GeoTIFF or PNG → (H, W, 3) float32 in [0, 1]."""
        path_str = str(path)

        if path_str.endswith((".tif", ".tiff")):
            try:
                with rasterio.open(path_str) as src:
                    img = src.read()
                    img = img[:3].transpose(1, 2, 0)
                return _to_float32(img)
            except Exception as e:
                print(f"[WARNING] rasterio failed for {path_str}: {e}. Falling back to cv2.")

        img = cv2.imread(path_str, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.0