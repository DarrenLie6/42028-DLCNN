from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import rasterio
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Config ────────────────────────────────────────────────────────────────────
cfg       = OmegaConf.load("configs/train_config.yaml")
ROOT      = Path(cfg.data.root_dir)
SPLIT_DIR = Path(cfg.data.split_file_dir)
PRE_DIR   = ROOT / cfg.data.pre_event_dir
POST_DIR  = ROOT / cfg.data.post_event_dir
TGT_DIR   = ROOT / cfg.data.target_dir
OUT_DIR   = Path("reports/eda")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES  = {0: "Background", 1: "Intact", 2: "Minor", 3: "Major", 4: "Destroyed"}
LABEL_COLORS = {0: "#d3d3d3", 1: "#2ecc71", 2: "#f1c40f", 3: "#e67e22", 4: "#e74c3c"}
SAR_ONLY     = {"ukraine-conflict", "myanmar-hurricane", "mexico-hurricane"}

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_all_stems():
    stems = {}
    split_config = {
        "train": cfg.data.train_split,
        "val": cfg.data.val_split,
        "test": cfg.data.test_split,
    }
    for split, filename in split_config.items():
        f = SPLIT_DIR / filename
        stems[split] = [s.strip() for s in f.read_text().splitlines() if s.strip()]
    return stems

def get_event(stem):
    return stem.rsplit("_", 1)[0]


# ── EDA 1: Tile counts per event ──────────────────────────────────────────────
def eda_tile_counts(all_stems):
    print("\n[1/5] Tile counts per event...")
    event_counts = defaultdict(lambda: defaultdict(int))
    for split, stems in all_stems.items():
        for stem in stems:
            event_counts[get_event(stem)][split] += 1

    events   = sorted(event_counts.keys())
    train_c  = [event_counts[e]["train"] for e in events]
    val_c    = [event_counts[e]["val"]   for e in events]
    test_c   = [event_counts[e]["test"]  for e in events]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(events))
    w = 0.28
    ax.bar(x - w, train_c, w, label="Train", color="#3498db")
    ax.bar(x,      val_c,  w, label="Val",   color="#2ecc71")
    ax.bar(x + w,  test_c, w, label="Test",  color="#e67e22")
    ax.set_xticks(x)
    ax.set_xticklabels(events, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Tile count")
    ax.set_title("Tile Distribution per Event and Split")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "1_tile_counts.png", dpi=150)
    plt.close()
    print(f"   Saved → {OUT_DIR}/1_tile_counts.png")

    print(f"\n   {'Event':<30} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>7}")
    print("   " + "-"*55)
    for e in events:
        tr = event_counts[e]["train"]
        va = event_counts[e]["val"]
        te = event_counts[e]["test"]
        flag = " [SAR-only]" if e in SAR_ONLY else ""
        print(f"   {e:<30} {tr:>6} {va:>6} {te:>6} {tr+va+te:>7}{flag}")


# ── EDA 2: Label class distribution ──────────────────────────────────────────
def eda_label_distribution(all_stems):
    print("\n[2/5] Label class distribution...")
    global_counts   = defaultdict(int)
    event_pct       = defaultdict(lambda: defaultdict(float))
    tile_damage_pct = []

    all_stems_flat = [s for stems in all_stems.values() for s in stems]

    for stem in tqdm(all_stems_flat, desc="   Scanning labels", ncols=70):
        lbl_path = TGT_DIR / f"{stem}_building_damage.tif"
        if not lbl_path.exists():
            continue
        with rasterio.open(lbl_path) as src:
            arr = src.read(1)
        unique, counts = np.unique(arr, return_counts=True)
        total_px = arr.size
        for cls, cnt in zip(unique.tolist(), counts.tolist()):
            global_counts[cls] += cnt
        dmg_pct = float((arr >= 2).sum()) / total_px * 100
        tile_damage_pct.append(dmg_pct)

        event = get_event(stem)
        for cls, cnt in zip(unique.tolist(), counts.tolist()):
            event_pct[event][cls] += cnt

    # Normalise event_pct to percentages
    for event in event_pct:
        total = sum(event_pct[event].values())
        for cls in event_pct[event]:
            event_pct[event][cls] = event_pct[event][cls] / total * 100

    # ── Plot A: global pie ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    labels_present = sorted(global_counts.keys())
    sizes  = [global_counts[c] for c in labels_present]
    colors = [LABEL_COLORS[c] for c in labels_present]
    names  = [LABEL_NAMES[c]  for c in labels_present]
    axes[0].pie(sizes, labels=names, colors=colors, autopct="%1.1f%%", startangle=140)
    axes[0].set_title("Global Pixel Class Distribution")

    # ── Plot B: event stacked bar ────────────────────────────────────────────
    events  = sorted(event_pct.keys())
    classes = [0, 1, 2, 3, 4]
    bottom  = np.zeros(len(events))
    for cls in classes:
        vals = [event_pct[e].get(cls, 0) for e in events]
        axes[1].bar(events, vals, bottom=bottom,
                    color=LABEL_COLORS[cls], label=LABEL_NAMES[cls])
        bottom += np.array(vals)
    axes[1].set_xticklabels(events, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Pixel %")
    axes[1].set_title("Class Distribution per Event")
    axes[1].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "2_label_distribution.png", dpi=150)
    plt.close()

    # ── Plot C: per-tile damage % histogram ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(tile_damage_pct, bins=50, color="#e74c3c", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Damaged pixel % per tile (classes 2–4)")
    ax.set_ylabel("Number of tiles")
    ax.set_title("Per-Tile Damage Density Distribution")
    pct_zero = sum(1 for v in tile_damage_pct if v == 0) / len(tile_damage_pct) * 100
    ax.axvline(0.1, color="black", linestyle="--", alpha=0.5)
    ax.text(0.5, ax.get_ylim()[1]*0.9, f"{pct_zero:.1f}% tiles have 0% damage",
            fontsize=9, color="black")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "3_damage_density.png", dpi=150)
    plt.close()
    print(f"   Saved → {OUT_DIR}/2_label_distribution.png")
    print(f"   Saved → {OUT_DIR}/3_damage_density.png")

    print(f"\n   Global class counts:")
    total_px = sum(global_counts.values())
    for cls in sorted(global_counts):
        cnt = global_counts[cls]
        print(f"   Class {cls} ({LABEL_NAMES[cls]:10s}): {cnt:>12,}  ({cnt/total_px*100:.1f}%)")


# ── EDA 3: Image statistics ───────────────────────────────────────────────────
def eda_image_stats(all_stems):
    print("\n[3/5] Computing image statistics (sample of 500 tiles)...")
    stems_flat = [s for stems in all_stems.values() for s in stems]
    np.random.seed(42)
    sample = np.random.choice(stems_flat, size=min(500, len(stems_flat)), replace=False)

    opt_means, opt_stds = [], []
    sar_means, sar_stds = [], []

    for stem in tqdm(sample, desc="   Reading tiles", ncols=70):
        event    = get_event(stem)
        opt_path = PRE_DIR  / f"{stem}_pre_disaster.tif"
        sar_path = POST_DIR / f"{stem}_post_disaster.tif"

        if event not in SAR_ONLY and opt_path.exists():
            with rasterio.open(opt_path) as src:
                opt = src.read().astype(np.float32)   # (3, H, W)
            opt_means.append(opt.mean(axis=(1, 2)))
            opt_stds.append(opt.std(axis=(1, 2)))

        if sar_path.exists():
            with rasterio.open(sar_path) as src:
                sar = src.read(1).astype(np.float32)  # (H, W)
            sar_means.append(sar.mean())
            sar_stds.append(sar.std())

    opt_means = np.array(opt_means)   # (N, 3)
    opt_stds  = np.array(opt_stds)
    sar_means = np.array(sar_means)
    sar_stds  = np.array(sar_stds)

    print(f"\n   Optical (RGB) per-channel stats  [{len(opt_means)} tiles]:")
    ch_names = ["R", "G", "B"]
    for i, ch in enumerate(ch_names):
        print(f"     {ch}: mean={opt_means[:, i].mean():.2f}  std={opt_stds[:, i].mean():.2f}")

    print(f"\n   SAR stats  [{len(sar_means)} tiles]:")
    print(f"     mean={sar_means.mean():.4f}  std={sar_stds.mean():.4f}")

    # ── Histogram of channel distributions ───────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#95a5a6"]
    for i, (ch, col) in enumerate(zip(["R", "G", "B", "SAR"], colors)):
        ax = axes[i]
        if i < 3:
            ax.hist(opt_means[:, i], bins=40, color=col, edgecolor="white", alpha=0.85)
            ax.set_title(f"Optical {ch} — mean per tile")
        else:
            ax.hist(sar_means, bins=40, color=col, edgecolor="white", alpha=0.85)
            ax.set_title("SAR — mean per tile")
        ax.set_xlabel("Pixel value")
        ax.set_ylabel("Tile count")
    plt.suptitle("Per-tile Mean Pixel Value Distributions", y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "4_image_stats.png", dpi=150)
    plt.close()
    print(f"\n   Saved → {OUT_DIR}/4_image_stats.png")

    # Print recommended normalization values
    print(f"\n   Recommended normalization values for train_config.yaml:")
    opt_global_mean = opt_means.mean(axis=0).tolist()
    opt_global_std  = opt_stds.mean(axis=0).tolist()
    print(f"     optical_mean: {[round(v,2) for v in opt_global_mean]}")
    print(f"     optical_std : {[round(v,2) for v in opt_global_std]}")
    print(f"     sar_mean    : [{sar_means.mean():.4f}]")
    print(f"     sar_std     : [{sar_stds.mean():.4f}]")


# ── EDA 4: Sample visualisation ──────────────────────────────────────────────
def eda_sample_visualization(all_stems):
    print("\n[4/5] Generating sample tile visualizations...")
    events = sorted({get_event(s) for stems in all_stems.values() for s in stems})
    train_stems = all_stems["train"]

    # Pick one tile per event with dual modality
    selected = {}
    for stem in train_stems:
        e = get_event(stem)
        if e not in selected and e not in SAR_ONLY:
            opt_p = PRE_DIR  / f"{stem}_pre_disaster.tif"
            sar_p = POST_DIR / f"{stem}_post_disaster.tif"
            lbl_p = TGT_DIR  / f"{stem}_building_damage.tif"
            if opt_p.exists() and sar_p.exists() and lbl_p.exists():
                selected[e] = stem

    events_with_tiles = sorted(selected.keys())
    n = len(events_with_tiles)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, event in enumerate(events_with_tiles):
        stem  = selected[event]
        opt_p = PRE_DIR  / f"{stem}_pre_disaster.tif"
        sar_p = POST_DIR / f"{stem}_post_disaster.tif"
        lbl_p = TGT_DIR  / f"{stem}_building_damage.tif"

        with rasterio.open(opt_p) as src:
            opt = src.read([1, 2, 3]).transpose(1, 2, 0).astype(np.float32)
        with rasterio.open(sar_p) as src:
            sar = src.read(1).astype(np.float32)
        with rasterio.open(lbl_p) as src:
            lbl = src.read(1)

        # Normalise for display
        def norm(x):
            lo, hi = np.percentile(x, 2), np.percentile(x, 98)
            return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)

        axes[row, 0].imshow(norm(opt))
        axes[row, 0].set_title(f"{event}\nPre-event Optical", fontsize=8)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(norm(sar), cmap="gray")
        axes[row, 1].set_title("Post-event SAR", fontsize=8)
        axes[row, 1].axis("off")

        lbl_rgb = np.zeros((*lbl.shape, 3), dtype=np.float32)
        for cls, hex_col in LABEL_COLORS.items():
            r, g, b = tuple(int(hex_col[i:i+2], 16)/255 for i in (1, 3, 5))
            lbl_rgb[lbl == cls] = [r, g, b]
        axes[row, 2].imshow(lbl_rgb)
        axes[row, 2].set_title("Damage Label", fontsize=8)
        axes[row, 2].axis("off")

    # Legend
    patches = [mpatches.Patch(color=LABEL_COLORS[c], label=LABEL_NAMES[c]) for c in range(5)]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, 0), fontsize=9)
    plt.suptitle("Sample Tiles: Pre-event Optical | Post-event SAR | Damage Label",
                 fontsize=12, y=1.005)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "5_sample_tiles.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"   Saved → {OUT_DIR}/5_sample_tiles.png")


# ── EDA 5: SAR-only event check ───────────────────────────────────────────────
def eda_sar_only_summary(all_stems):
    print("\n[5/5] SAR-only event summary...")
    for split, stems in all_stems.items():
        total      = len(stems)
        sar_only_n = sum(1 for s in stems if get_event(s) in SAR_ONLY)
        pct        = sar_only_n / total * 100
        print(f"   {split:5s}: {sar_only_n:>4} / {total} SAR-only tiles ({pct:.1f}%)")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("BRIGHT DATASET — EXPLORATORY DATA ANALYSIS")
    print("="*60)

    all_stems = load_all_stems()

    eda_tile_counts(all_stems)
    eda_label_distribution(all_stems)
    eda_image_stats(all_stems)
    eda_sample_visualization(all_stems)
    eda_sar_only_summary(all_stems)

    print("\n" + "="*60)
    print(f"EDA complete! All plots saved to: {OUT_DIR}/")
    print("="*60 + "\n")