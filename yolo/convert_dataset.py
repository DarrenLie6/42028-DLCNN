from __future__ import annotations
import cv2
import numpy as np
import rasterio
import yaml
from pathlib import Path


# Config
DATA_ROOT  = Path("42028-DLCNN/data/bright")
IMAGE_DIR  = DATA_ROOT / "post-event"
LABEL_DIR  = DATA_ROOT / "target"
SPLITS_DIR = DATA_ROOT / "splits"
OUT_ROOT   = Path("bright_yolo")

CLASS_MAP = {
    0: 1,  # intact
    1: 2,  # damaged
    2: 3,  # destroyed
}

MIN_AREA = {
    0: 15,  # intact
    1: 20,  # damaged
    2: 20,  # destroyed
}

SPLITS = {
    "train": "train_set.txt",
    "val":   "val_set.txt",
    "test":  "test_set.txt",
}


def convert_image_for_yolo(image_path: Path, out_path: Path) -> None:
    with rasterio.open(image_path) as src:
        img = src.read(1).astype(np.float32)

    # SAR backscatter has a heavy tail so log-compress it first,
    # then clip outliers and stretch to [0, 1].
    # Matches load_sar() in normalization_utils.py exactly.
    img = np.log1p(img)
    p2, p98 = np.percentile(img, 2), np.percentile(img, 98)
    img = np.clip(img, p2, p98)
    img = (img - p2) / ((p98 - p2) + 1e-8)

    # YOLO expects 3-channel images, so stack the single SAR band 3 times
    img_uint8 = (img * 255).astype(np.uint8)
    img_3ch   = np.stack([img_uint8, img_uint8, img_uint8], axis=-1)
    cv2.imwrite(str(out_path), img_3ch)


def mask_to_yolo_polygons(
    label_path: Path,
    class_map: dict[int, int],
    min_area: dict[int, int],
) -> list[str]:
    with rasterio.open(label_path) as src:
        mask = src.read(1)

    H, W = mask.shape
    lines: list[str] = []

    for yolo_cls, pixel_val in class_map.items():
        # Isolate this class as a binary mask
        binary = np.uint8(mask == pixel_val) * 255

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            if cv2.contourArea(cnt) < min_area[yolo_cls]:
                continue

            # Reduce point count while keeping the shape accurate
            epsilon = 0.002 * cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, epsilon, True)

            if len(cnt) < 3:
                continue

            # YOLO stores coordinates as fractions of image size, not pixels
            pts = cnt.reshape(-1, 2).astype(float)
            pts[:, 0] /= W
            pts[:, 1] /= H

            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
            lines.append(f"{yolo_cls} {coords}")

    return lines


def convert_split(split_name: str, split_filename: str) -> dict[int, int]:
    stems = (SPLITS_DIR / split_filename).read_text().strip().splitlines()

    img_out = OUT_ROOT / "images" / split_name
    lbl_out = OUT_ROOT / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    print(f"[{split_name}] converting {len(stems)} tiles")

    class_counts = {0: 0, 1: 0, 2: 0}
    skipped = 0

    for stem in stems:
        img_path = IMAGE_DIR / f"{stem}_post_disaster.tif"
        lbl_path = LABEL_DIR / f"{stem}_building_damage.tif"

        if not img_path.exists() or not lbl_path.exists():
            print(f"missing files for {stem}, skipping")
            skipped += 1
            continue

        convert_image_for_yolo(img_path, img_out / f"{stem}.png")

        lines = mask_to_yolo_polygons(lbl_path, CLASS_MAP, MIN_AREA)
        for line in lines:
            class_counts[int(line.split()[0])] += 1

        with open(lbl_out / f"{stem}.txt", "w") as f:
            f.write("\n".join(lines))

    if skipped:
        print(f"skipped {skipped} tiles with missing files")
    print(
        f"intact: {class_counts[0]:,}  "
        f"damaged: {class_counts[1]:,}  "
        f"destroyed: {class_counts[2]:,}"
    )
    return class_counts


def write_data_yaml() -> None:
    cfg = {
        "path":  str(OUT_ROOT.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    3,
        "names": ["intact", "damaged", "destroyed"],
    }
    out = OUT_ROOT / "data.yaml"
    with open(out, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"data.yaml written to {out}")


def sanity_check() -> None:
    sample = next((OUT_ROOT / "labels" / "train").glob("*.txt"), None)
    if sample is None:
        print("no label files found in labels/train - check paths")
        return
    lines = sample.read_text().strip().splitlines()
    print(f"sanity check on {sample.name}")
    print(f"  instances: {len(lines)}")
    if lines:
        print(f"  first line: {lines[0][:80]}{'...' if len(lines[0]) > 80 else ''}")
    else:
        print("empty - tile has no foreground buildings")


if __name__ == "__main__":
    total_counts = {0: 0, 1: 0, 2: 0}

    for split_name, split_file in SPLITS.items():
        counts = convert_split(split_name, split_file)
        for k in total_counts:
            total_counts[k] += counts[k]

    write_data_yaml()
    sanity_check()

    print("\ndone")
    print(
        f"total — "
        f"intact: {total_counts[0]:,}  "
        f"damaged: {total_counts[1]:,}  "
        f"destroyed: {total_counts[2]:,}"
    )