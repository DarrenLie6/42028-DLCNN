"""
Evaluation script for semantic segmentation on xView2.

Computes per-class and overall metrics:
  - IoU (Intersection over Union)
  - F1 score
  - Precision & Recall
  - Confusion matrix
  - Per-class accuracy

Also saves visual sample comparisons:
  - Post-disaster image | Ground truth overlay | Prediction overlay
"""

from __future__ import annotations
import argparse
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from tqdm import tqdm

from src.models.deeplabv3 import build_semantic_model, SemanticSegmentationXViewDataset
from src.training.metrics import SegmentationMetrics


NUM_CLASSES  = 4
IGNORE_INDEX = -100
LABEL_NAMES  = {0: "Background", 1: "Intact", 2: "Damaged", 3: "Destroyed"}

# Colour palette for overlay — RGBA in [0, 1]
# Background is fully transparent so the image shows through
CLASS_COLOURS = {
    0: (0.00, 0.00, 0.00, 0.00),   # Background  — transparent
    1: (0.18, 0.80, 0.44, 0.55),   # Intact      — green
    2: (1.00, 0.75, 0.00, 0.65),   # Damaged     — amber
    3: (0.90, 0.18, 0.18, 0.75),   # Destroyed   — red
}

NUM_VISUAL_SAMPLES = 12   # how many comparison grids to save


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    print(f"[Loading] {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = build_semantic_model(
        num_classes=NUM_CLASSES,
        pretrained=False,
        device=device,
    )

    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    if missing:
        print(f"  [WARN] Missing keys:    {missing}")
    if unexpected:
        print(f"  [WARN] Unexpected keys: {unexpected}")

    model.eval()

    epoch    = checkpoint.get("epoch", "?")
    mean_iou = checkpoint.get("mean_iou", "?")
    print(f"  Epoch: {epoch} | Mean IoU: {mean_iou}")

    return model


# ------------------------------------------------------------------
# Visual overlay helpers
# ------------------------------------------------------------------

def mask_to_rgba(mask: np.ndarray) -> np.ndarray:
    """
    Convert a (H, W) integer class mask to an (H, W, 4) RGBA overlay.
    Background pixels are transparent; building pixels are coloured.
    """
    h, w   = mask.shape
    rgba   = np.zeros((h, w, 4), dtype=np.float32)
    for cls, colour in CLASS_COLOURS.items():
        rgba[mask == cls] = colour
    return rgba


def save_sample_comparisons(
    predictions: list,
    save_dir: str,
    num_samples: int = NUM_VISUAL_SAMPLES,
) -> None:
    """
    Save side-by-side comparison figures:
        Post-disaster image | Ground truth overlay | Prediction overlay

    Picks samples that have at least one non-background pixel in the GT
    so the comparisons are visually informative.
    """
    vis_dir = Path(save_dir) / "sample_comparisons"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Prefer samples with actual building annotations
    annotated = [p for p in predictions if (p["gt"] > 0).any()]
    unannotated = [p for p in predictions if not (p["gt"] > 0).any()]
    ordered = annotated + unannotated

    samples = ordered[:num_samples]

    legend_patches = [
        mpatches.Patch(color=CLASS_COLOURS[i][:3], alpha=0.8, label=LABEL_NAMES[i])
        for i in range(1, NUM_CLASSES)   # skip background
    ]

    for idx, pred in enumerate(samples):
        image  = pred["image"]    # (H, W, 3) float32 [0,1]
        gt     = pred["gt"]       # (H, W) int
        preds  = pred["pred"]     # (H, W) int
        stem   = pred["stem"]

        gt_rgba   = mask_to_rgba(gt)
        pred_rgba = mask_to_rgba(preds)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.patch.set_facecolor("#111111")

        titles = ["Post-Disaster Image", "Ground Truth", "Prediction"]
        for ax, title in zip(axes, titles):
            ax.set_facecolor("#111111")
            ax.set_title(title, color="white", fontsize=12, pad=8)
            ax.axis("off")

        # Col 1 — raw image
        axes[0].imshow(image)

        # Col 2 — image + GT overlay
        axes[1].imshow(image)
        axes[1].imshow(gt_rgba)

        # Col 3 — image + prediction overlay
        axes[2].imshow(image)
        axes[2].imshow(pred_rgba)

        # Shared legend at the bottom
        fig.legend(
            handles=legend_patches,
            loc="lower center",
            ncol=3,
            framealpha=0.2,
            labelcolor="white",
            fontsize=10,
            bbox_to_anchor=(0.5, -0.04),
        )

        fig.suptitle(
            f"Sample: {stem}",
            color="white", fontsize=11, y=1.01,
        )

        plt.tight_layout(rect=[0, 0.04, 1, 1])

        out_path = vis_dir / f"sample_{idx:03d}_{stem[:40]}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)

    print(f"✓ {len(samples)} sample comparisons saved → {vis_dir}")


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    save_dir: str = "evaluation_results",
) -> dict:
    """Run evaluation and save metrics + visual samples."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    model.eval()
    metrics = SegmentationMetrics(
        num_classes=NUM_CLASSES,
        ignore_index=IGNORE_INDEX,
        device=device,
    )

    all_predictions = []

    print("\n[Evaluation] Running inference...")
    with torch.no_grad():
        for batch in tqdm(data_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            stems  = batch["stem"]

            outputs = model(images)
            logits  = outputs["out"]

            metrics.update(logits, labels)

            preds = logits.argmax(dim=1)  # (B, H, W)

            for i, stem in enumerate(stems):
                # Convert image tensor → (H, W, 3) numpy for visualisation.
                # Bi-temporal inputs are 6ch [pre|post]; show the post half.
                img_t = images[i]
                if img_t.shape[0] == 6:
                    img_t = img_t[3:]
                img_np = img_t.cpu().numpy().transpose(1, 2, 0)
                img_np = np.clip(img_np, 0.0, 1.0)

                all_predictions.append({
                    "stem":   stem,
                    "image":  img_np,
                    "pred":   preds[i].cpu().numpy(),
                    "gt":     labels[i].cpu().numpy(),
                    "logits": logits[i].cpu().numpy(),
                })

    results = metrics.compute()

    print("\n[Results] Per-Class Metrics:")
    print("-" * 60)
    print(f"{'Class':<15} {'IoU':>10} {'F1':>10} {'Acc':>10}")
    print("-" * 60)

    for class_idx, class_name in LABEL_NAMES.items():
        iou = results.get(f"iou/{class_name}", 0.0)
        f1  = results.get(f"f1/{class_name}",  0.0)
        acc = results.get(f"acc/{class_name}", 0.0)
        print(f"{class_name:<15} {iou:>10.4f} {f1:>10.4f} {acc:>10.4f}")

    print("-" * 60)
    print(
        f"{'Mean':<15} {results['mean_iou']:>10.4f} "
        f"{results['mean_f1']:>10.4f} {results['mean_acc']:>10.4f}"
    )
    print("-" * 60)

    save_results(results, all_predictions, save_dir)

    return results


# ------------------------------------------------------------------
# Saving
# ------------------------------------------------------------------

def save_results(results: dict, predictions: list, save_dir: str) -> None:
    """Save metrics JSON, plots, and visual sample comparisons."""

    # Metrics JSON
    metrics_json = {
        k: float(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v
        for k, v in results.items()
    }
    json_path = Path(save_dir) / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"\n✓ Metrics saved to {json_path}")

    plot_confusion_matrix(predictions, save_dir)
    plot_per_class_metrics(results, save_dir)

    # Visual comparisons — post-disaster image vs GT vs prediction
    save_sample_comparisons(predictions, save_dir, num_samples=NUM_VISUAL_SAMPLES)

    # Per-sample pixel accuracy
    pred_summary_path = Path(save_dir) / "predictions_summary.txt"
    with open(pred_summary_path, "w") as f:
        f.write("Per-Sample IoU Analysis\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Stem':<30} {'IoU':>10}\n")
        f.write("-" * 60 + "\n")

        sample_ious = []
        for pred in predictions:
            pred_mask = pred["pred"]
            gt_mask   = pred["gt"]
            intersection = (pred_mask == gt_mask).sum()
            sample_iou   = intersection / pred_mask.size if pred_mask.size > 0 else 0.0
            sample_ious.append(sample_iou)
            f.write(f"{pred['stem']:<30} {sample_iou:>10.4f}\n")

        f.write("-" * 60 + "\n")
        f.write(f"{'Mean':<30} {np.mean(sample_ious):>10.4f}\n")
        f.write(f"{'Std':<30}  {np.std(sample_ious):>10.4f}\n")

    print(f"✓ Per-sample predictions saved to {pred_summary_path}")


def plot_confusion_matrix(predictions: list, save_dir: str) -> None:
    """Plot and save confusion matrix."""
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for pred in predictions:
        pred_mask = pred["pred"].reshape(-1)
        gt_mask   = pred["gt"].reshape(-1)
        valid     = gt_mask != IGNORE_INDEX
        pred_mask = pred_mask[valid]
        gt_mask   = gt_mask[valid]
        for p, g in zip(pred_mask, gt_mask):
            if 0 <= p < NUM_CLASSES and 0 <= g < NUM_CLASSES:
                cm[g, p] += 1

    class_names = [LABEL_NAMES[i] for i in range(NUM_CLASSES)]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, cbar_kws={"label": "Count"},
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Ground Truth Label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    cm_path = Path(save_dir) / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Confusion matrix saved to {cm_path}")


def plot_per_class_metrics(results: dict, save_dir: str) -> None:
    """Plot per-class IoU, F1, and Accuracy bar charts."""
    classes = list(LABEL_NAMES.values())
    ious    = [results.get(f"iou/{c}", 0.0) for c in classes]
    f1s     = [results.get(f"f1/{c}",  0.0) for c in classes]
    accs    = [results.get(f"acc/{c}", 0.0) for c in classes]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, values, color, ylabel, title, mean_key in zip(
        axes,
        [ious,         f1s,        accs],
        ["steelblue",  "seagreen", "coral"],
        ["IoU",        "F1 Score", "Accuracy"],
        ["Intersection over Union", "F1 Score", "Per-Class Accuracy"],
        ["mean_iou",   "mean_f1",  "mean_acc"],
    ):
        ax.bar(classes, values, color=color)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim([0, 1])
        mean_val = results.get(mean_key, 0.0)
        ax.axhline(y=mean_val, color="r", linestyle="--",
                   label=f"Mean: {mean_val:.3f}")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    metrics_path = Path(save_dir) / "per_class_metrics.png"
    plt.savefig(metrics_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Per-class metrics plot saved to {metrics_path}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main(
    checkpoint_path: str,
    config_path: str,
    split: str = "val",
    output_dir: str = "evaluation_results",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")

    cfg   = OmegaConf.load(config_path)
    model = load_checkpoint(checkpoint_path, device)

    print(f"\n[Data] Building {split} dataset...")
    dataset = SemanticSegmentationXViewDataset(
        root_dir=cfg.data.root_dir,
        cfg=cfg,
        mode=split,
        transform=None,
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )

    results = evaluate(model, loader, device, output_dir)

    print(f"\n✓ Evaluation complete!")
    print(f"  Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate semantic segmentation model on xView2"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config",     type=str, default="configs/deeplabv3_config.yaml")
    parser.add_argument("--split",      type=str, default="val",
                        choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=str, default="evaluation_results")

    args = parser.parse_args()
    main(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        split=args.split,
        output_dir=args.output_dir,
    )