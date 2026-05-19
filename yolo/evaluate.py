"""YOLO11 evaluation and visualization script."""

import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from ultralytics import YOLO
from config import (
    YOLO_CLASSES, YOLO_DS_DIR, IMG_SIZE, EVAL_CONFIG, INFERENCE_CONFIG
)


def load_best_model(run_name="xbd_seg_v1"):
    """Load the best trained model weights."""
    best_weights = os.path.join(YOLO_DS_DIR, "runs", run_name, "weights", "best.pt")
    if not os.path.exists(best_weights):
        raise FileNotFoundError(f"Best weights not found: {best_weights}")
    
    model = YOLO(best_weights)
    print(f"Loaded model: {best_weights}")
    return model


def run_inference(model, split="val", num_samples=50):
    """Run inference on a dataset split."""
    val_img_dir = os.path.join(YOLO_DS_DIR, "images", split)
    images = sorted(glob.glob(os.path.join(val_img_dir, "*.png")))[:num_samples]
    
    print(f"\nRunning inference on {len(images)} {split} images...")
    pred_results = model.predict(source=images, **INFERENCE_CONFIG)
    print(f"Inference complete on {len(pred_results)} images.")
    
    return pred_results


def build_damage_heatmap(result, img_size=IMG_SIZE):
    """
    From a YOLO result, build a damage heatmap accumulator.
    
    Returns:
        heatmaps: dict {class_idx: (H,W) float32 accumulation}
        orig_img: (H,W,3) uint8 original image
    """
    h, w = img_size, img_size
    heatmaps = {
        0: np.zeros((h, w), dtype=np.float32),
        1: np.zeros((h, w), dtype=np.float32),
        2: np.zeros((h, w), dtype=np.float32)
    }
    
    orig_img = result.orig_img  # BGR, HxWx3
    
    if result.masks is None:
        return heatmaps, cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    
    masks = result.masks.data.cpu().numpy()   # (N, H, W) float32 [0,1]
    class_ids = result.boxes.cls.cpu().numpy().astype(int)  # (N,)
    confs = result.boxes.conf.cpu().numpy()             # (N,)
    
    for mask, cls_id, conf in zip(masks, class_ids, confs):
        # Resize mask to output size if needed
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        # Weight mask by confidence score
        heatmaps[cls_id] += mask * float(conf)
    
    return heatmaps, cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)


def render_heatmap_figure(orig_img, heatmaps, title="Damage Heatmap"):
    """
    Renders a 5-panel figure:
      [original | intact heatmap | damaged heatmap | destroyed heatmap | composite]
    """
    h, w = orig_img.shape[:2]
    
    # Composite overlay: blend all class heatmaps onto image
    composite = orig_img.copy().astype(np.float32)
    yolo_colors = {
        0: np.array([0,   200,  0],  dtype=np.float32),  # intact
        1: np.array([255, 165,  0],  dtype=np.float32),  # damaged
        2: np.array([220,  0,   0],  dtype=np.float32),  # destroyed
    }
    
    for cls_id, color in yolo_colors.items():
        hm = heatmaps[cls_id]
        if hm.max() > 0:
            norm_hm = (hm / hm.max())[:, :, np.newaxis]  # (H,W,1)
            composite = composite * (1 - 0.6 * norm_hm) + color * (0.6 * norm_hm)
    composite = np.clip(composite, 0, 255).astype(np.uint8)
    
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(title, fontsize=13, fontweight='bold')
    
    axes[0].imshow(orig_img)
    axes[0].set_title("Post-disaster Image")
    axes[0].axis("off")
    
    for i, (cls_id, cmap, label) in enumerate([
        (0, 'Greens',  'Intact'),
        (1, 'Oranges', 'Damaged'),
        (2, 'Reds',    'Destroyed'),
    ]):
        hm = heatmaps[cls_id]
        im = axes[i+1].imshow(hm, cmap=cmap, vmin=0, vmax=max(hm.max(), 1e-6))
        axes[i+1].imshow(orig_img, alpha=0.3)   # ghost image underneath
        axes[i+1].set_title(f"{label} Heatmap")
        axes[i+1].axis("off")
        plt.colorbar(im, ax=axes[i+1], fraction=0.046, pad=0.04)
    
    axes[4].imshow(composite)
    axes[4].set_title("Composite Overlay")
    axes[4].axis("off")
    
    # Legend
    patches = [
        mpatches.Patch(color='green',  label='Intact'),
        mpatches.Patch(color='orange', label='Damaged'),
        mpatches.Patch(color='red',    label='Destroyed'),
    ]
    axes[4].legend(handles=patches, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    return fig


def save_heatmaps(pred_results, num_samples=6, output_dir=None):
    """Render and save heatmaps for sample predictions."""
    if output_dir is None:
        output_dir = os.path.join(YOLO_DS_DIR, "heatmaps")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating heatmaps for {num_samples} samples...")
    for i, result in enumerate(pred_results[:num_samples]):
        heatmaps, orig_img = build_damage_heatmap(result, img_size=IMG_SIZE)
        scene_name = os.path.basename(result.path).replace('.png', '')
        fig = render_heatmap_figure(orig_img, heatmaps, title=f"Scene: {scene_name}")
        save_path = os.path.join(output_dir, f"{scene_name}_heatmap.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved → {save_path}")


def aggregate_disaster_heatmap(pred_results, output_dir=None, img_size=IMG_SIZE):
    """
    Accumulate heatmaps across ALL predictions to show overall spatial
    damage distribution for the dataset.
    """
    if output_dir is None:
        output_dir = os.path.join(YOLO_DS_DIR, "heatmaps")
    os.makedirs(output_dir, exist_ok=True)
    
    agg = {
        0: np.zeros((img_size, img_size), dtype=np.float32),
        1: np.zeros((img_size, img_size), dtype=np.float32),
        2: np.zeros((img_size, img_size), dtype=np.float32)
    }
    
    print("\nAggregating heatmaps across all predictions...")
    for result in tqdm(pred_results, desc="Aggregating"):
        heatmaps, _ = build_damage_heatmap(result, img_size)
        for cls_id in agg:
            agg[cls_id] += heatmaps[cls_id]
    
    # Normalize
    for cls_id in agg:
        if agg[cls_id].max() > 0:
            agg[cls_id] /= agg[cls_id].max()
    
    # Plot aggregate
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(f"Aggregate Damage Heatmap — {len(pred_results)} Scenes", fontsize=13)
    
    titles = ['Intact',  'Damaged', 'Destroyed', 'Severity\n(Destroyed − Intact)']
    cmaps  = ['Greens',  'Oranges', 'Reds',      'RdYlGn_r']
    data   = [agg[0],    agg[1],    agg[2],
              np.clip(agg[2] - agg[0], 0, 1)]  # net severity score
    
    for ax, title, cmap, d in zip(axes, titles, cmaps, data):
        im = ax.imshow(d, cmap=cmap, vmin=0, vmax=1, interpolation='bilinear')
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    agg_path = os.path.join(output_dir, "aggregate_heatmap.png")
    fig.savefig(agg_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Aggregate heatmap saved → {agg_path}")


def evaluate_model(model, yaml_path):
    """Run official YOLO validation (gives mAP50, mAP50-95, per-class metrics)."""
    print(f"\n{'='*70}")
    print("Running Official Validation")
    print(f"{'='*70}\n")
    
    config = EVAL_CONFIG.copy()
    config['data'] = yaml_path
    metrics = model.val(**config)
    
    print(f"\n{'='*70}")
    print("Segmentation Metrics")
    print(f"{'='*70}")
    print(f"  mAP@50      : {metrics.seg.map50:.4f}")
    print(f"  mAP@50-95   : {metrics.seg.map:.4f}")
    print(f"\nPer-class mAP@50:")
    for cls_name, ap in zip(YOLO_CLASSES, metrics.seg.ap50):
        print(f"  {cls_name:<12s}  {ap:.4f}")
    print(f"{'='*70}\n")
    
    return metrics


def main():
    """Main evaluation pipeline."""
    # Load best model
    model = load_best_model()
    
    # Run inference on validation set
    pred_results = run_inference(model, split="val", num_samples=50)
    
    # Generate heatmaps
    save_heatmaps(pred_results, num_samples=6)
    aggregate_disaster_heatmap(pred_results)
    
    # Evaluate on full validation set
    yaml_path = os.path.join(YOLO_DS_DIR, "xbd.yaml")
    if os.path.exists(yaml_path):
        evaluate_model(model, yaml_path)
    else:
        print(f"Warning: Dataset YAML not found at {yaml_path}")
        print("Run train.py first to generate the YAML file.")


if __name__ == "__main__":
    main()
