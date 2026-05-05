"""
evaluate.py
===========
Evaluates a trained YOLOv8-seg model on the BRIGHT test set using the
same metrics as the UNet baseline (metrics.py):

    Per-class IoU, F1, Accuracy for: Intact / Damaged / Destroyed
    Mean IoU, Mean F1, Mean Accuracy (foreground classes only, bg excluded)

Instance masks are converted back to a semantic segmentation map so results
are directly comparable to the UNet's pixel-level predictions.

When two instance masks overlap, the higher damage class wins (conservative
damage assessment — better to overestimate than miss damage).

Usage:
    python yolo/evaluate.py
    python yolo/evaluate.py --weights path/to/best.pt --split val
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
import torch
import yaml
from ultralytics import YOLO


# Config
DEFAULT_WEIGHTS = "runs/segment/yolo_bright/yolov8m_seg_sar/weights/best.pt"
DATA_YAML       = "bright_yolo/data.yaml"
BRIGHT_ROOT     = Path("42028-DLCNN/data/bright")

NUM_CLASSES  = 4
IGNORE_INDEX = 0 
LABEL_NAMES  = {0: "Background", 1: "Intact", 2: "Damaged", 3: "Destroyed"}

# YOLO class id (0-indexed) -> BRIGHT label value (1-indexed)
YOLO_TO_BRIGHT = {0: 1, 1: 2, 2: 3}


def masks_to_semantic(result, img_h: int, img_w: int) -> np.ndarray:
    """
    Convert YOLO instance segmentation output -> semantic map (H, W) int64.

    Pixel values match BRIGHT label convention:
        0 = background
        1 = intact
        2 = damaged
        3 = destroyed

    Overlap resolution: higher damage class wins.
    This is conservative — avoids under-reporting damage in dense scenes.
    """
    semantic = np.zeros((img_h, img_w), dtype=np.int64)

    if result.masks is None:
        return semantic  # no detections → all background

    # Resize masks to original image size if needed
    masks   = result.masks.data.cpu().numpy()   # (N, H', W') float32
    classes = result.boxes.cls.cpu().numpy()    # (N,) float32

    for mask, cls in zip(masks, classes):
        bright_cls = YOLO_TO_BRIGHT[int(cls)]
        binary     = mask > 0.5  # threshold to bool

        # Resize mask to match original image if YOLO downsampled it
        if binary.shape != (img_h, img_w):
            import cv2
            binary = cv2.resize(
                binary.astype(np.uint8),
                (img_w, img_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        # Higher damage class wins on overlap
        semantic[binary] = np.maximum(semantic[binary], bright_cls)

    return semantic


def load_gt_label(stem: str) -> np.ndarray:
    """Load ground-truth BRIGHT label TIF → (H, W) int64."""
    lbl_path = BRIGHT_ROOT / "target" / f"{stem}_building_damage.tif"
    if not lbl_path.exists():
        return None
    with rasterio.open(lbl_path) as src:
        return src.read(1).astype(np.int64)


def update_confusion_matrix(
    conf_matrix: torch.Tensor,
    pred: np.ndarray,
    gt: np.ndarray,
) -> None:
    """Accumulate one tile into the confusion matrix. Matches metrics.py logic."""
    valid_mask   = gt != IGNORE_INDEX
    preds_flat   = torch.tensor(pred[valid_mask],   dtype=torch.long)
    targets_flat = torch.tensor(gt[valid_mask],     dtype=torch.long)

    # Clamp predictions to valid range (YOLO might occasionally go out of range)
    preds_flat = preds_flat.clamp(0, NUM_CLASSES - 1)

    indices = targets_flat * NUM_CLASSES + preds_flat
    conf_matrix += torch.bincount(
        indices, minlength=NUM_CLASSES ** 2
    ).reshape(NUM_CLASSES, NUM_CLASSES)


def compute_metrics(conf_matrix: torch.Tensor) -> dict[str, float]:
    """
    Compute per-class IoU, F1, Accuracy from confusion matrix.
    Identical to metrics.py SegmentationMetrics.compute().
    """
    cm  = conf_matrix.float()
    tp  = cm.diag()
    fp  = cm.sum(dim=0) - tp
    fn  = cm.sum(dim=1) - tp
    eps = 1e-7

    iou = tp / (tp + fp + fn + eps)
    f1  = (2 * tp) / (2 * tp + fp + fn + eps)
    acc = tp / (cm.sum(dim=1) + eps)

    results: dict[str, float] = {}
    fg = [i for i in LABEL_NAMES if i != IGNORE_INDEX]

    for idx, name in LABEL_NAMES.items():
        results[f"iou/{name}"] = iou[idx].item()
        results[f"f1/{name}"]  = f1[idx].item()
        results[f"acc/{name}"] = acc[idx].item()

    results["mean_iou"] = iou[fg].mean().item()
    results["mean_f1"]  = f1[fg].mean().item()
    results["mean_acc"] = acc[fg].mean().item()

    return results


def print_results(results: dict[str, float]) -> None:
    print("\n" + "=" * 55)
    print("  YOLO Evaluation Results (matched to UNet metrics)")
    print("=" * 55)
    print(f"  {'Class':<12}  {'IoU':>7}  {'F1':>7}  {'Acc':>7}")
    print("  " + "-" * 40)
    for idx, name in LABEL_NAMES.items():
        if idx == IGNORE_INDEX:
            continue
        iou = results[f"iou/{name}"]
        f1  = results[f"f1/{name}"]
        acc = results[f"acc/{name}"]
        print(f"  {name:<12}  {iou:>7.4f}  {f1:>7.4f}  {acc:>7.4f}")
    print("  " + "-" * 40)
    print(f"  {'Mean (fg)':<12}  {results['mean_iou']:>7.4f}  "
          f"{results['mean_f1']:>7.4f}  {results['mean_acc']:>7.4f}")
    print("=" * 55)


def evaluate(weights: str, data_yaml: str, split: str = "test") -> dict[str, float]:
    # Load dataset config
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    data_root = Path(cfg["path"])
    img_dir   = data_root / "images" / split

    image_paths = sorted(img_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG images found in {img_dir}")

    print(f"Evaluating {len(image_paths)} tiles from '{split}' split ...")
    print(f"Weights: {weights}\n")

    model = YOLO(weights)
    conf_matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)

    missing_gt = 0
    for img_path in image_paths:
        stem = img_path.stem

        # Ground truth from original BRIGHT TIF (more reliable than re-parsing YOLO txt)
        gt = load_gt_label(stem)
        if gt is None:
            missing_gt += 1
            continue

        # YOLO inference
        result = model(str(img_path), verbose=False)[0]
        H, W   = result.orig_shape

        # Instance masks → semantic map
        pred = masks_to_semantic(result, H, W)

        # Accumulate confusion matrix
        update_confusion_matrix(conf_matrix, pred, gt)

    if missing_gt > 0:
        print(f"⚠  Skipped {missing_gt} tiles with missing GT labels")

    results = compute_metrics(conf_matrix)
    print_results(results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate YOLOv8-seg on BRIGHT dataset"
    )
    parser.add_argument(
        "--weights", default=DEFAULT_WEIGHTS,
        help=f"Path to model weights (default: {DEFAULT_WEIGHTS})"
    )
    parser.add_argument(
        "--data", default=DATA_YAML,
        help=f"Path to data.yaml (default: {DATA_YAML})"
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "val", "test"],
        help="Which split to evaluate on (default: test)"
    )
    args = parser.parse_args()

    evaluate(
        weights   = args.weights,
        data_yaml = args.data,
        split     = args.split,
    )