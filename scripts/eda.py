from __future__ import annotations

import sys
import json
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import rasterio
from rasterio.transform import Affine
from omegaconf import OmegaConf
from tqdm import tqdm
from shapely.geometry import shape

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Config ────────────────────────────────────────────────────────────────────
cfg       = OmegaConf.load("configs/train_config.yaml")
ROOT      = Path(cfg.data.root_dir)
OUT_DIR   = Path("reports/eda")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# xview2 dataset uses tier1/tier3 directories instead of separate pre/post/target
DATA_TIERS = [ROOT / "tier1", ROOT / "tier3"]
TEST_DIR   = ROOT / "test"
HOLD_DIR   = ROOT / "hold"

LABEL_NAMES  = {0: "Background", 1: "Intact", 2: "Damaged", 3: "Destroyed"}
LABEL_COLORS = {0: "#d3d3d3", 1: "#2ecc71", 2: "#e67e22", 3: "#e74c3c"}

# Mapping from xview2 JSON damage subtypes to numeric classes
DAMAGE_CLASS_MAPPING = {
    "no-damage": 1,        # Intact
    "minor-damage": 2,     # Damaged
    "major-damage": 2,     # Damaged
    "destroyed": 3,        # Destroyed
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_all_stems():
    """Load stems from xview2 split files (train_set.txt, val_set.txt)."""
    stems = {}
    
    # Load train and val splits
    train_file = ROOT / "train_set.txt"
    val_file = ROOT / "val_set.txt"
    
    if train_file.exists():
        stems["train"] = [s.strip() for s in train_file.read_text().splitlines() if s.strip()]
    else:
        stems["train"] = []
        
    if val_file.exists():
        stems["val"] = [s.strip() for s in val_file.read_text().splitlines() if s.strip()]
    else:
        stems["val"] = []
    
    # For xview2, discover test files from test/ directory if it exists
    if TEST_DIR.exists() and (TEST_DIR / "images").exists():
        test_images = list((TEST_DIR / "images").glob("*_pre_disaster.tif"))
        stems["test"] = [img.stem.replace("_pre_disaster", "") for img in test_images]
    else:
        stems["test"] = []
    
    return stems

def get_event(stem):
    """Extract event name from stem. e.g., 'guatemala-volcano_00000000' -> 'guatemala-volcano'."""
    return stem.rsplit("_", 1)[0]

def find_image_pair(stem):
    """Find pre and post disaster image files for a given stem.
    
    Returns: (pre_path, post_path, tier_dir) or (None, None, None) if not found
    """
    # Check tier1 and tier3 first
    for tier_dir in DATA_TIERS:
        img_dir = tier_dir / "images"
        pre_path = img_dir / f"{stem}_pre_disaster.tif"
        post_path = img_dir / f"{stem}_post_disaster.tif"
        
        if pre_path.exists() and post_path.exists():
            return pre_path, post_path, tier_dir
    
    # Check test and hold directories
    for tier_dir in [TEST_DIR, HOLD_DIR]:
        if not tier_dir.exists():
            continue
        img_dir = tier_dir / "images"
        pre_path = img_dir / f"{stem}_pre_disaster.tif"
        post_path = img_dir / f"{stem}_post_disaster.tif"
        
        if pre_path.exists() and post_path.exists():
            return pre_path, post_path, tier_dir
    
    return None, None, None

def load_json_labels(stem, tier_dir):
    """Load damage labels from xview2 JSON label file and rasterize to pixel coordinates.
    
    Returns tuple: (damage_counts_dict, damage_mask_array) or (None, None) if not found
    """
    lbl_dir = tier_dir / "labels"
    post_label_path = lbl_dir / f"{stem}_post_disaster.json"
    img_path = tier_dir / "images" / f"{stem}_post_disaster.tif"
    
    if not post_label_path.exists():
        return None, None
    
    try:
        with open(post_label_path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"   Error loading {post_label_path}: {e}")
        return None, None
    
    # Get image dimensions and transform
    try:
        with rasterio.open(img_path) as src:
            h, w = src.height, src.width
            transform = src.transform
    except:
        return None, None
    
    # Extract damage classes from features
    damage_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    damage_mask = np.zeros((h, w), dtype=np.uint8)
    
    features = data.get("features", {})
    
    # Process features - use only lng_lat coordinates (geographic)
    # Note: xy coordinates are already in pixel space but may have different origin/scale
    if "lng_lat" in features:
        for building in features["lng_lat"]:
            if "wkt" not in building:
                continue
            
            try:
                # Parse WKT geometry
                wkt_str = building.get("wkt", "")
                if not wkt_str.startswith("POLYGON"):
                    continue
                
                # Extract coordinates from WKT: POLYGON ((lon1 lat1, lon2 lat2, ...))
                coords_str = wkt_str.replace("POLYGON", "").replace("(", "").replace(")", "").strip()
                lon_lat_coords = []
                for coord_pair in coords_str.split(","):
                    parts = coord_pair.strip().split()
                    if len(parts) >= 2:
                        try:
                            lon, lat = float(parts[0]), float(parts[1])
                            lon_lat_coords.append((lon, lat))
                        except:
                            pass
                
                if len(lon_lat_coords) < 3:
                    continue
                
                # Convert from lon/lat to pixel coordinates using rasterio transform
                pixel_coords = []
                for lon, lat in lon_lat_coords:
                    # Get pixel row, col from geographic coordinates
                    col = (lon - transform.c) / transform.a
                    row = (lat - transform.f) / transform.e
                    
                    # Clip to image bounds
                    col = max(0, min(w - 1, col))
                    row = max(0, min(h - 1, row))
                    pixel_coords.append([int(col), int(row)])
                
                if len(pixel_coords) < 3:
                    continue
                
                pixel_coords = np.array(pixel_coords, dtype=np.int32)
                
                # Rasterize the polygon - get damage class from properties
                props = building.get("properties", {})
                subtype = props.get("subtype", "no-damage")
                damage_class = DAMAGE_CLASS_MAPPING.get(subtype, 1)
                damage_counts[damage_class] += 1
                
                # Fill polygon on mask
                cv2.fillPoly(damage_mask, [pixel_coords], damage_class)
                
            except Exception as e:
                continue
    
    # Count background pixels
    total_pixels = h * w
    building_pixels = (damage_mask > 0).sum()
    damage_counts[0] = total_pixels - building_pixels  # Background
    
    total_buildings = sum([damage_counts[c] for c in [1, 2, 3]])
    if total_buildings == 0:
        return None, None
    
    return damage_counts, damage_mask


# ── EDA 1: Tile counts per event ──────────────────────────────────────────────
def eda_tile_counts(all_stems):
    print("\n[1/5] Tile counts per event...")
    event_counts = defaultdict(lambda: defaultdict(int))
    
    # Count available tiles (those with both pre and post images)
    for split, stems in all_stems.items():
        for stem in stems:
            pre, post, tier = find_image_pair(stem)
            if pre and post:  # Only count if both images exist
                event_counts[get_event(stem)][split] += 1

    events   = sorted(event_counts.keys())
    train_c  = [event_counts[e].get("train", 0) for e in events]
    val_c    = [event_counts[e].get("val", 0) for e in events]
    test_c   = [event_counts[e].get("test", 0) for e in events]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(events))
    w = 0.28
    ax.bar(x - w, train_c, w, label="Train", color="#3498db")
    ax.bar(x,      val_c,  w, label="Val",   color="#2ecc71")
    ax.bar(x + w,  test_c, w, label="Test",  color="#e67e22")
    ax.set_xticks(x)
    ax.set_xticklabels(events, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Tile count")
    ax.set_title("Tile Distribution per Event and Split (xview2)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "1_tile_counts.png", dpi=150)
    plt.close()
    print(f"   Saved → {OUT_DIR}/1_tile_counts.png")

    print(f"\n   {'Event':<30} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>7}")
    print("   " + "-"*55)
    for e in events:
        tr = event_counts[e].get("train", 0)
        va = event_counts[e].get("val", 0)
        te = event_counts[e].get("test", 0)
        print(f"   {e:<30} {tr:>6} {va:>6} {te:>6} {tr+va+te:>7}")


# ── EDA 2: Label class distribution ──────────────────────────────────────────
def eda_label_distribution(all_stems):
    print("\n[2/5] Label class distribution (from JSON annotations)...")
    global_counts   = defaultdict(int)
    event_pct       = defaultdict(lambda: defaultdict(float))
    building_damage_pct = []

    all_stems_flat = [s for stems in all_stems.values() for s in stems]

    for stem in tqdm(all_stems_flat, desc="   Scanning labels", ncols=70):
        pre, post, tier = find_image_pair(stem)
        if not (pre and post):
            continue
            
        damage_counts, damage_mask = load_json_labels(stem, tier)
        if damage_counts is None or damage_mask is None:
            continue
        
        # Add to global counts (including background class 0)
        for cls in [0, 1, 2, 3]:
            global_counts[cls] += damage_counts.get(cls, 0)
        
        # Calculate damage percentage (classes 2, 3 are damaged/destroyed)
        total_buildings = sum([damage_counts.get(c, 0) for c in [1, 2, 3]])
        if total_buildings > 0:
            dmg_count = damage_counts.get(2, 0) + damage_counts.get(3, 0)
            dmg_pct = (dmg_count / total_buildings * 100)
            building_damage_pct.append(dmg_pct)
        
        # Add to event percentages (only buildings, not background)
        event = get_event(stem)
        for cls in [1, 2, 3]:
            event_pct[event][cls] += damage_counts.get(cls, 0)

    # Normalise event_pct to percentages
    for event in event_pct:
        total = sum(event_pct[event].values())
        if total > 0:
            for cls in event_pct[event]:
                event_pct[event][cls] = event_pct[event][cls] / total * 100

    # ── Plot A: global pixel distribution (includes background) ──────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: All pixels including background
    classes_all = [c for c in [0, 1, 2, 3] if c in global_counts and global_counts[c] > 0]
    sizes_all = [global_counts[c] for c in classes_all]
    colors_all = [LABEL_COLORS.get(c, "#cccccc") for c in classes_all]
    names_all = [LABEL_NAMES.get(c, "Background") for c in classes_all]
    
    if sizes_all:
        axes[0].pie(sizes_all, labels=names_all, colors=colors_all, autopct="%1.1f%%", startangle=140)
        axes[0].set_title("Global Pixel Distribution\n(all pixels including background)")
    else:
        axes[0].text(0.5, 0.5, "No labels found", ha='center', va='center')
    
    # Right: Only building classes (1, 2, 3) for clearer view of damage distribution
    classes_bldg = [c for c in [1, 2, 3] if c in global_counts and global_counts[c] > 0]
    sizes_bldg = [global_counts[c] for c in classes_bldg]
    colors_bldg = [LABEL_COLORS.get(c, "#cccccc") for c in classes_bldg]
    names_bldg = [LABEL_NAMES.get(c, "Building") for c in classes_bldg]
    
    if sizes_bldg:
        axes[1].pie(sizes_bldg, labels=names_bldg, colors=colors_bldg, autopct="%1.1f%%", startangle=140)
        axes[1].set_title("Building-Only Distribution\n(damage classes only)")
    else:
        axes[1].text(0.5, 0.5, "No buildings found", ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "2_label_distribution.png", dpi=150)
    plt.close()
    print(f"   Saved → {OUT_DIR}/2_label_distribution.png")

    # ── Plot B: event stacked bar ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    events  = sorted(event_pct.keys())
    if events:
        classes = [1, 2, 3]
        bottom  = np.zeros(len(events))
        for cls in classes:
            vals = [event_pct[e].get(cls, 0) for e in events]
            ax.bar(events, vals, bottom=bottom,
                        color=LABEL_COLORS[cls], label=LABEL_NAMES[cls])
            bottom += np.array(vals)
        ax.set_xticklabels(events, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Building %")
        ax.set_title("Damage Class Distribution per Event (building pixels only)")
        ax.legend(loc="upper right", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "2b_event_distribution.png", dpi=150)
    plt.close()
    print(f"   Saved → {OUT_DIR}/2b_event_distribution.png")

    # ── Plot C: per-tile damage % histogram ──────────────────────────────────
    if building_damage_pct:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(building_damage_pct, bins=50, color="#e74c3c", edgecolor="white", alpha=0.8)
        ax.set_xlabel("Damaged/Destroyed building % per tile (classes 2–3)")
        ax.set_ylabel("Number of tiles")
        ax.set_title("Per-Tile Damage Density Distribution (xview2)")
        pct_zero = sum(1 for v in building_damage_pct if v < 1) / len(building_damage_pct) * 100
        ax.axvline(1.0, color="black", linestyle="--", alpha=0.5, label="1% threshold")
        ax.text(ax.get_xlim()[1]*0.5, ax.get_ylim()[1]*0.9, f"{pct_zero:.1f}% tiles have <1% damage",
                fontsize=9, color="black")
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / "3_damage_density.png", dpi=150)
        plt.close()
        print(f"   Saved → {OUT_DIR}/3_damage_density.png")

    print(f"\n   Global pixel distribution:")
    total_pixels = sum([global_counts[c] for c in [0, 1, 2, 3]])
    if total_pixels > 0:
        print(f"   Class 0 (Background):       {global_counts[0]:>12,}  ({global_counts[0]/total_pixels*100:.1f}%)")
        for cls in [1, 2, 3]:
            cnt = global_counts[cls]
            pct = (cnt / total_pixels * 100) if total_pixels > 0 else 0
            print(f"   Class {cls} ({LABEL_NAMES[cls]:10s}): {cnt:>12,}  ({pct:.1f}%)")
        
        print(f"\n   Building-only damage distribution:")
        total_buildings = sum([global_counts[c] for c in [1, 2, 3]])
        if total_buildings > 0:
            for cls in [1, 2, 3]:
                cnt = global_counts[cls]
                pct = (cnt / total_buildings * 100) if total_buildings > 0 else 0
                print(f"   Class {cls} ({LABEL_NAMES[cls]:10s}): {cnt:>12,}  ({pct:.1f}%)")


# ── EDA 3: Image statistics ───────────────────────────────────────────────────
def eda_image_stats(all_stems):
    print("\n[3/5] Computing image statistics (sample of 500 tiles)...")
    stems_flat = [s for stems in all_stems.values() for s in stems]
    np.random.seed(42)
    sample = np.random.choice(stems_flat, size=min(500, len(stems_flat)), replace=False)

    opt_means, opt_stds = [], []

    for stem in tqdm(sample, desc="   Reading tiles", ncols=70):
        pre, post, tier = find_image_pair(stem)
        if not (pre and post):
            continue

        # xview2 pre-disaster images are RGB (optical)
        with rasterio.open(pre) as src:
            opt = src.read().astype(np.float32)   # (3, H, W)
        if opt.shape[0] >= 3:
            opt = opt[:3]  # Take first 3 channels if more than 3
            opt_means.append(opt.mean(axis=(1, 2)))
            opt_stds.append(opt.std(axis=(1, 2)))

    opt_means = np.array(opt_means) if opt_means else np.zeros((0, 3))
    opt_stds  = np.array(opt_stds) if opt_stds else np.zeros((0, 3))

    print(f"\n   Optical (RGB) per-channel stats  [{len(opt_means)} tiles]:")
    ch_names = ["R", "G", "B"]
    if len(opt_means) > 0:
        for i, ch in enumerate(ch_names):
            print(f"     {ch}: mean={opt_means[:, i].mean():.2f}  std={opt_stds[:, i].mean():.2f}")

    # ── Histogram of channel distributions ───────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    for i, (ch, col) in enumerate(zip(["R", "G", "B"], colors)):
        ax = axes[i]
        if len(opt_means) > 0:
            ax.hist(opt_means[:, i], bins=40, color=col, edgecolor="white", alpha=0.85)
            ax.set_title(f"Optical {ch} — mean per tile")
        ax.set_xlabel("Pixel value")
        ax.set_ylabel("Tile count")
    plt.suptitle("Per-tile Mean Pixel Value Distributions (xview2)", y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "4_image_stats.png", dpi=150)
    plt.close()
    print(f"\n   Saved → {OUT_DIR}/4_image_stats.png")

    # Print recommended normalization values (normalized to [0,1] range)
    print(f"\n   Recommended normalization values for train_config.yaml:")
    if len(opt_means) > 0:
        opt_global_mean = (opt_means.mean(axis=0) / 255.0).tolist()
        opt_global_std  = (opt_stds.mean(axis=0) / 255.0).tolist()
        print(f"     optical_mean: {[round(v,3) for v in opt_global_mean]}")
        print(f"     optical_std : {[round(v,3) for v in opt_global_std]}")
    else:
        print("     (No optical images found)")


# ── EDA 4: Sample visualisation ──────────────────────────────────────────────
def eda_sample_visualization(all_stems):
    print("\n[4/5] Generating sample tile visualizations (pre | post | damage mask)...")
    events = sorted({get_event(s) for stems in all_stems.values() for s in stems})
    
    # Collect all splits for better sampling
    all_stems_list = []
    for split in ["train", "val", "test"]:
        all_stems_list.extend(all_stems.get(split, []))
    
    # Pick one tile per event with labels
    selected = {}
    for stem in all_stems_list:
        e = get_event(stem)
        if e not in selected:
            pre, post, tier = find_image_pair(stem)
            if pre and post:
                damage_counts, damage_mask = load_json_labels(stem, tier)
                if damage_counts and damage_mask is not None:
                    selected[e] = (stem, tier)

    events_with_tiles = sorted(selected.keys())
    n = len(events_with_tiles)
    
    if n == 0:
        print("   No tiles with labels found for visualization")
        return
    
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, event in enumerate(events_with_tiles):
        stem, tier = selected[event]
        pre, post, _ = find_image_pair(stem)
        damage_counts, damage_mask = load_json_labels(stem, tier)

        # Read pre and post disaster images
        with rasterio.open(pre) as src:
            pre_data = src.read([1, 2, 3]).transpose(1, 2, 0).astype(np.float32)
            
        with rasterio.open(post) as src:
            post_data = src.read([1, 2, 3]).transpose(1, 2, 0).astype(np.float32)

        # Normalise for display
        def norm(x):
            lo, hi = np.percentile(x, 2), np.percentile(x, 98)
            return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)

        pre_norm = norm(pre_data)
        post_norm = norm(post_data)

        axes[row, 0].imshow(pre_norm)
        axes[row, 0].set_title(f"{event}\nPre-disaster", fontsize=9)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(post_norm)
        axes[row, 1].set_title("Post-disaster", fontsize=9)
        axes[row, 1].axis("off")

        # Overlay damage mask on post-disaster image
        h, w = damage_mask.shape
        damage_overlay = post_norm.copy()
        
        # Create colored overlay for each damage class
        # Only render if mask has non-zero values
        if damage_mask.max() > 0:
            for cls in [3, 2, 1]:  # Draw in reverse order so lower classes visible
                mask_class = (damage_mask == cls)
                if mask_class.any():
                    r, g, b = tuple(int(LABEL_COLORS[cls][i:i+2], 16)/255 for i in (1, 3, 5))
                    # Alpha blend: 0.6 opacity for the overlay
                    alpha = 0.6
                    damage_overlay[mask_class] = alpha * np.array([r, g, b]) + (1 - alpha) * post_norm[mask_class]
        
        axes[row, 2].imshow(damage_overlay)
        
        # Show statistics in title
        background = damage_counts.get(0, 0)
        intact = damage_counts.get(1, 0)
        damaged = damage_counts.get(2, 0)
        destroyed = damage_counts.get(3, 0)
        total_pixels = h * w
        total_buildings = intact + damaged + destroyed
        
        # Calculate percentages
        pct_background = background / total_pixels * 100
        pct_buildings = total_buildings / total_pixels * 100
        pct_dmg = (damaged + destroyed) / total_buildings * 100 if total_buildings > 0 else 0
        
        axes[row, 2].set_title(f"Damage Overlay\n(BG: {pct_background:.0f}% | {total_buildings} buildings: {pct_dmg:.0f}% damaged)", fontsize=8)
        axes[row, 2].axis("off")

    # Legend
    patches = [mpatches.Patch(color=LABEL_COLORS[c], label=LABEL_NAMES[c]) for c in [1, 2, 3]]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.01), fontsize=10)
    plt.suptitle("Sample Tiles: Pre-disaster | Post-disaster | Damage Labels (from polygon annotations)",
                 fontsize=12, y=1.001)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "5_sample_tiles.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"   Saved → {OUT_DIR}/5_sample_tiles.png")


# ── EDA 5: Dataset split summary ──────────────────────────────────────────────
def eda_split_summary(all_stems):
    print("\n[5/5] Dataset split summary...")
    for split, stems in all_stems.items():
        available = sum(1 for s in stems if find_image_pair(s)[0] is not None)
        total = len(stems)
        pct = (available / total * 100) if total > 0 else 0
        print(f"   {split:5s}: {available:>4} / {total} tiles available ({pct:.1f}%)")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("XVIEW2 DATASET — EXPLORATORY DATA ANALYSIS")
    print("="*60)

    all_stems = load_all_stems()

    eda_tile_counts(all_stems)
    eda_label_distribution(all_stems)
    eda_image_stats(all_stems)
    eda_sample_visualization(all_stems)
    eda_split_summary(all_stems)

    print("\n" + "="*60)
    print(f"EDA complete! All plots saved to: {OUT_DIR}/")
    print("="*60 + "\n")