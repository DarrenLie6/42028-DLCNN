"""
verify_xview2_conversion.py

Sanity check script to verify xView2 -> BRIGHT conversion was successful.
Checks that converted files exist, SAR is zero-filled, and labels have
valid damage classes with non-zero building pixels.

Usage:
    # From your project root (42028-DLCNN/)
    python scripts/verify_xview2_conversion.py

    # Optionally point to a different config
    python scripts/verify_xview2_conversion.py --config configs/train_config.yaml

    # Check more tiles
    python scripts/verify_xview2_conversion.py --n 20
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
import yaml


# Known xView2 disaster event name prefixes
XVIEW2_KEYWORDS = [
    "guatemala", "joplin", "santa-rosa", "moore", "nepal",
    "pinery", "portugal", "socal", "tuscaloosa", "hurricane",
    "midwest", "mexico", "woolsey", "lower-puna", "midwest",
    "harvey", "matthew", "michael", "florence", "palu",
]


def load_cfg_paths(config_path: Path) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    data = cfg["data"]
    return {
        "root_dir":      Path(data["root_dir"]),
        "pre_event_dir": data["pre_event_dir"],
        "post_event_dir": data["post_event_dir"],
        "target_dir":    data["target_dir"],
        "split_file_dir": Path(data["split_file_dir"]),
        "train_split":   data["train_split"],
    }


def verify(config_path: Path, n: int):
    paths = load_cfg_paths(config_path)
    bright = paths["root_dir"]
    train_txt = paths["split_file_dir"] / paths["train_split"]

    if not train_txt.exists():
        print(f"[ERROR] train_set.txt not found at {train_txt}")
        return

    all_stems = train_txt.read_text().splitlines()
    xview2_stems = [
        s for s in all_stems
        if any(k in s for k in XVIEW2_KEYWORDS)
    ]

    print(f"Total stems in train_set.txt : {len(all_stems)}")
    print(f"xView2 stems found           : {len(xview2_stems)}")

    if not xview2_stems:
        print("\n[WARN] No xView2 stems found in train_set.txt.")
        print("       Run convert_xview2_to_bright.py first.")
        return

    check_stems = xview2_stems[:n]
    print(f"Checking first {len(check_stems)} xView2 stems...\n")

    passed = 0
    all_zero_labels = 0
    missing_files = 0

    for stem in check_stems:
        print(f"--- {stem} ---")
        pre = bright / paths["pre_event_dir"]  / f"{stem}_pre_disaster.tif"
        sar = bright / paths["post_event_dir"] / f"{stem}_post_disaster.tif"
        lbl = bright / paths["target_dir"]     / f"{stem}_building_damage.tif"

        stem_ok = True

        for name, path in [("pre", pre), ("sar", sar), ("lbl", lbl)]:
            if not path.exists():
                print(f"  {name}: MISSING ❌")
                missing_files += 1
                stem_ok = False
                continue
            with rasterio.open(path) as src:
                arr = src.read()
                print(f"  {name}: shape={arr.shape} dtype={arr.dtype} unique={np.unique(arr).tolist()[:10]}{'...' if len(np.unique(arr)) > 10 else ''}")

        if not stem_ok:
            continue

        # SAR should be all zeros
        with rasterio.open(sar) as src:
            sar_arr = src.read()
            if sar_arr.sum() == 0:
                print(f"  SAR zero-check  : PASSED ✅")
            else:
                print(f"  SAR zero-check  : FAILED ❌ (non-zero values found)")
                stem_ok = False

        # Label should have valid classes and non-zero building pixels
        with rasterio.open(lbl) as src:
            lbl_arr = src.read(1)
            bad_classes = set(np.unique(lbl_arr)) - {0, 1, 2, 3}
            n_building = (lbl_arr > 0).sum()

            if bad_classes:
                print(f"  Label class-check: FAILED (unexpected classes: {bad_classes})")
                stem_ok = False
            elif n_building == 0:
                print(f"  Label class-check: WARNING  (no building pixels — tile may be empty)")
                all_zero_labels += 1
            else:
                print(f"  Label class-check: PASSED ({n_building} building pixels)")

        if stem_ok and n_building > 0:
            passed += 1

        print()

    print("=" * 50)
    print(f"Results: {passed}/{len(check_stems)} fully passed")
    if all_zero_labels:
        print(f"   {all_zero_labels} tiles had no building pixels (may be valid empty tiles)")
    if missing_files:
        print(f"  {missing_files} missing files detected")
    if passed == len(check_stems):
        print("  All checks passed — ready to train!")


def main():
    parser = argparse.ArgumentParser(description="Verify xView2 -> BRIGHT conversion")
    parser.add_argument(
        "--config",
        default="configs/train_config.yaml",
        help="Path to train_config.yaml (default: configs/train_config.yaml)"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of xView2 stems to check (default: 5)"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    verify(config_path, args.n)


if __name__ == "__main__":
    main()