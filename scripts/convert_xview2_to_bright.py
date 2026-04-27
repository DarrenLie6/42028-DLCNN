"""
convert_xview2_to_bright.py

Converts xView2 dataset into BRIGHT format and appends all xView2 stems
to bright's train_set.txt. Reads output paths from your train_config.yaml
so there is no hardcoding.

Usage:
    # From your project root (42028-DLCNN/)
    python convert_xview2_to_bright.py \
        --xview2 /Volumes/SSD/xview2/xview2

    # Optionally point to a different config
    python convert_xview2_to_bright.py \
        --xview2 /Volumes/SSD/xview2/xview2 \
        --config configs/train_config.yaml

What it does:
    - Reads bright paths from train_config.yaml (root_dir, pre_event_dir, etc.)
    - Copies xView2 pre-event TIFs -> bright/pre-event/{stem}_pre_disaster.tif
    - Writes zero-filled SAR placeholders -> bright/post-event/{stem}_post_disaster.tif
    - Rasterizes xView2 GeoJSON labels -> bright/target/{stem}_building_damage.tif
      with class remapping: 0=background, 1=no-damage, 2=minor, 3=major, 4=destroyed->3
    - Appends all xView2 stems to bright/splits/train_set.txt (no duplicates)

Splits included: tier1, tier3, hold
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.features import rasterize
from shapely.wkt import loads as wkt_loads
from shapely.geometry import mapping
from tqdm import tqdm


# xView2 subtype string -> bright label int
# bright: 0=background, 1=no-damage, 2=minor, 3=major/destroyed
SUBTYPE_TO_CLASS = {
    "no-damage":    1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed":    3,
    "un-classified": 1,
}


def load_cfg_paths(config_path: Path) -> dict:
    """Read only the data paths we need from train_config.yaml."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    data = cfg["data"]
    return {
        "root_dir":       Path(data["root_dir"]),
        "pre_event_dir":  data["pre_event_dir"],
        "post_event_dir": data["post_event_dir"],
        "target_dir":     data["target_dir"],
        "split_file_dir": Path(data["split_file_dir"]),
        "train_split":    data["train_split"],
    }


def copy_tif(src_path: Path, dst_path: Path, n_bands: int = 3):
    """Copy a TIF preserving geospatial metadata, limiting to n_bands."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_path) as src:
        arr = src.read()[:n_bands]
        profile = src.profile.copy()
        profile.update(count=arr.shape[0], compress="lzw", nodata=None)
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(arr)


def write_zero_sar(dst_path: Path, ref_path: Path):
    """Write a zero-filled single-band TIF matching ref_path's spatial profile."""
    with rasterio.open(ref_path) as src:
        profile = src.profile.copy()
        h, w = src.height, src.width

    arr = np.zeros((1, h, w), dtype=np.uint8)
    profile.update(count=1, dtype=np.uint8, compress="lzw", nodata=None)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(arr)


def rasterize_json_label(json_path: Path, ref_tif_path: Path, dst_path: Path):
    """
    Rasterize an xView2 GeoJSON label file into a single-band uint8 TIF.
    Reads polygon geometry + subtype, burns damage class values into a mask
    matching the spatial extent of ref_tif_path.
    """
    with open(json_path) as f:
        data = json.load(f)

    with rasterio.open(ref_tif_path) as src:
        profile = src.profile.copy()
        h, w = src.height, src.width

    # xView2 WKT coords are pixel (x, y) — use identity transform so
    # rasterize treats them as direct pixel coordinates, not geographic
    from rasterio.transform import Affine
    identity = Affine(1, 0, 0, 0, 1, 0)

    shapes = []
    for feat in data.get("features", {}).get("xy", []):
        subtype = feat.get("properties", {}).get("subtype", "no-damage")
        cls = SUBTYPE_TO_CLASS.get(subtype, 1)
        wkt = feat.get("wkt", "")
        if not wkt:
            continue
        try:
            geom = wkt_loads(wkt)
            if geom.is_empty:
                continue
            shapes.append((mapping(geom), cls))
        except Exception:
            continue

    if shapes:
        mask = rasterize(
            shapes,
            out_shape=(h, w),
            transform=identity,
            fill=0,
            dtype=np.uint8,
            all_touched=False,
        )
    else:
        mask = np.zeros((h, w), dtype=np.uint8)

    profile.update(count=1, dtype=np.uint8, compress="lzw", nodata=None)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(mask[np.newaxis, :, :])


def convert(paths: dict, xview2_root: Path):
    root          = paths["root_dir"]
    pre_event_dir = root / paths["pre_event_dir"]
    post_event_dir= root / paths["post_event_dir"]
    target_dir    = root / paths["target_dir"]
    splits_dir    = paths["split_file_dir"]
    train_txt     = splits_dir / paths["train_split"]

    for d in [pre_event_dir, post_event_dir, target_dir, splits_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load existing train stems to avoid duplicates
    existing_stems = set()
    if train_txt.exists():
        existing_stems = set(train_txt.read_text().splitlines())
        print(f"[INFO] Existing train stems: {len(existing_stems)}")

    new_stems = []
    skipped = 0
    label_missing = 0

    for split in ["tier1", "tier3", "hold"]:
        images_dir = xview2_root / split / "images"
        labels_dir = xview2_root / split / "labels"

        if not images_dir.exists():
            print(f"[WARN] {images_dir} not found - skipping split '{split}'")
            continue

        pre_files = sorted(images_dir.glob("*_pre_disaster.tif"))
        print(f"\n[{split}] Found {len(pre_files)} tiles")

        for pre_tif in tqdm(pre_files, desc=f"  Converting {split}"):
            stem = pre_tif.stem.replace("_pre_disaster", "")

            if stem in existing_stems:
                skipped += 1
                continue

            label_json = labels_dir / f"{stem}_post_disaster.json"

            # 1. Pre-event optical -> pre-event/
            copy_tif(pre_tif, pre_event_dir / f"{stem}_pre_disaster.tif", n_bands=3)

            # 2. Zero SAR placeholder -> post-event/
            write_zero_sar(post_event_dir / f"{stem}_post_disaster.tif", ref_path=pre_tif)

            # 3. Rasterize JSON label -> target/
            dst_lbl = target_dir / f"{stem}_building_damage.tif"
            if label_json.exists():
                rasterize_json_label(label_json, ref_tif_path=pre_tif, dst_path=dst_lbl)
            else:
                print(f"  [WARN] No label JSON for {stem} - writing zeros")
                write_zero_sar(dst_lbl, ref_path=pre_tif)
                label_missing += 1

            new_stems.append(stem)
            existing_stems.add(stem)

    # Append new stems to train_set.txt
    if new_stems:
        with open(train_txt, "a") as f:
            f.write("\n" + "\n".join(new_stems))
        print(f"\n[DONE] Appended {len(new_stems)} new xView2 stems to {train_txt}")
    else:
        print("\n[DONE] No new stems to add.")

    if skipped:
        print(f"[INFO] Skipped {skipped} stems already in train_set.txt")
    if label_missing:
        print(f"[WARN] {label_missing} tiles had no label JSON - written as zeros")
    print(f"[INFO] Total train stems now: {len(existing_stems)}")


def main():
    parser = argparse.ArgumentParser(description="Convert xView2 -> BRIGHT format")
    parser.add_argument(
        "--xview2",
        required=True,
        help="Path to xView2 dataset root (containing tier1, tier3, hold, test)"
    )
    parser.add_argument(
        "--config",
        default="configs/train_config.yaml",
        help="Path to train_config.yaml (default: configs/train_config.yaml)"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    xview2_root = Path(args.xview2)

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not xview2_root.exists():
        raise FileNotFoundError(f"xView2 root not found: {xview2_root}")

    paths = load_cfg_paths(config_path)

    print(f"Config      : {config_path}")
    print(f"BRIGHT root : {paths['root_dir']}")
    print(f"xView2 root : {xview2_root}")
    convert(paths, xview2_root)


if __name__ == "__main__":
    main()