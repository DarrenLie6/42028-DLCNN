"""
Comprehensive model evaluation — xBD / xView2 Siamese UNet.
Evaluates all 5 classes including background.
Only pixels labelled -100 (truly unlabelled) are excluded.
Samples are selected randomly across the test set.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import rasterio
import cv2
import random
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────
IGNORE_INDEX = -100
NUM_CLASSES  = 5

LABEL_NAMES = {
    0: "Un-classified",
    1: "No-Damage",
    2: "Minor-Damage",
    3: "Major-Damage",
    4: "Destroyed",
}

PALETTE = {
    0: (0.15, 0.15, 0.15),   # Un-classified — dark grey
    1: (0.20, 0.60, 0.20),   # No-Damage     — green
    2: (1.00, 1.00, 0.00),   # Minor-Damage  — yellow
    3: (1.00, 0.55, 0.00),   # Major-Damage  — orange
    4: (0.85, 0.15, 0.15),   # Destroyed     — red
}


# ── Helpers ───────────────────────────────────────────────────────────

def labels_to_rgb(mask: np.ndarray) -> np.ndarray:
    """(H,W) int mask → (H,W,3) RGB float32."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.float32)
    for cls, colour in PALETTE.items():
        rgb[mask == cls] = colour
    return rgb


def _load_display_image(path) -> np.ndarray:
    """
    Load raw image from disk for display purposes only.
    Applies per-channel percentile stretch so dark GeoTIFFs are visible.
    Returns (H,W,3) float32 in [0,1].
    """
    path = Path(path)
    try:
        with rasterio.open(str(path)) as src:
            img = src.read()[:3].transpose(1, 2, 0).astype(np.float32)
    except Exception:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            return np.zeros((256, 256, 3), dtype=np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        return np.clip(img / 255.0, 0, 1)

    # Percentile stretch per channel — handles uint8, uint16, float equally
    for c in range(3):
        ch      = img[:, :, c]
        p2, p98 = np.percentile(ch, 2), np.percentile(ch, 98)
        if p98 > p2:
            img[:, :, c] = (ch - p2) / (p98 - p2 + 1e-8)

    return np.clip(img, 0, 1)


def _find_stem_folder(dataset, stem: str) -> str:
    """Find which folder (tier1/tier3/hold/test) a stem belongs to."""
    for folder, s in dataset.stems:
        if s == stem:
            return folder
    raise ValueError(f"Stem not found in dataset: {stem}")


# ── Evaluator ─────────────────────────────────────────────────────────

class ModelEvaluator:
    """
    Evaluate a Siamese UNet on the xBD test set.
    - All 5 classes (0=Un-classified through 4=Destroyed) are included
    - Only pixels labelled -100 (truly unlabelled) are excluded
    - Samples for visualisation are chosen randomly across the test set
    """

    def __init__(
        self,
        model,
        test_loader: DataLoader,
        device:      torch.device,
        save_dir:    str = "eval_results",
        num_samples: int = 5,
        seed:        int = 42,
    ):
        self.model       = model.to(device).eval()
        self.test_loader = test_loader
        self.device      = device
        self.save_dir    = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.num_samples = num_samples
        self.seed        = seed

        self.all_preds   = []
        self.all_targets = []

        # Reservoir for random sample selection
        self._reservoir  = []   # list of candidate dicts
        self._seen       = 0    # total tiles seen so far

        random.seed(self.seed)

    # ── Inference ─────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self):
        """Run inference on full test set, accumulate predictions and random samples."""
        print("\n[Evaluation] Running inference on test set...")
        dataset = self.test_loader.dataset

        for batch in tqdm(self.test_loader, desc="Evaluating"):
            pre    = batch["pre_disaster"].to(self.device)
            post   = batch["post_disaster"].to(self.device)
            labels = batch["label"].to(self.device)
            stems  = batch["stem"]

            logits    = self.model(pre, post)
            preds     = logits.argmax(dim=1)

            preds_np  = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()

            # Accumulate metrics — exclude only -100 pixels
            valid = labels_np != IGNORE_INDEX
            self.all_preds.extend(preds_np[valid].tolist())
            self.all_targets.extend(labels_np[valid].tolist())

            # ── Reservoir sampling for random visualisation picks ──────
            for i in range(len(stems)):
                self._seen += 1
                candidate = {
                    "stem":   stems[i],
                    "pred":   preds_np[i],
                    "target": labels_np[i],
                }
                if len(self._reservoir) < self.num_samples:
                    self._reservoir.append(candidate)
                else:
                    # Replace a random existing entry with decreasing probability
                    j = random.randint(0, self._seen - 1)
                    if j < self.num_samples:
                        self._reservoir[j] = candidate

        self.all_preds   = np.array(self.all_preds,   dtype=np.int64)
        self.all_targets = np.array(self.all_targets, dtype=np.int64)
        print(f"[Evaluation] Valid pixels evaluated : {len(self.all_preds):,}")
        print(f"[Evaluation] Random samples selected: {len(self._reservoir)}")

        # Load display images from disk for the selected samples
        self.samples = []
        for c in self._reservoir:
            stem   = c["stem"]
            folder = _find_stem_folder(dataset, stem)
            img_dir = dataset.root / folder / "images"

            pre_path  = dataset._find_image(img_dir, f"{stem}_pre_disaster")
            post_path = dataset._find_image(img_dir, f"{stem}_post_disaster")

            self.samples.append({
                "stem":   stem,
                "pre":    _load_display_image(pre_path),
                "post":   _load_display_image(post_path),
                "pred":   c["pred"],
                "target": c["target"],
            })

    # ── Confusion Matrix ──────────────────────────────────────────────

    def compute_confusion_matrix(self):
        """Compute and save both raw and normalised confusion matrices."""
        print("\n[Evaluation] Computing confusion matrix...")

        cm = confusion_matrix(
            self.all_targets,
            self.all_preds,
            labels=list(range(NUM_CLASSES))
        )

        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm  = np.divide(
            cm.astype(float), row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=row_sums != 0
        )

        class_labels = [LABEL_NAMES[i] for i in range(NUM_CLASSES)]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels,
            cbar_kws={"label": "Count"}, ax=ax1
        )
        ax1.set_title("Confusion Matrix (Raw Counts)", fontsize=13, fontweight="bold")
        ax1.set_ylabel("True Label", fontsize=11)
        ax1.set_xlabel("Predicted Label", fontsize=11)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha="right")

        sns.heatmap(
            cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels,
            vmin=0, vmax=1, cbar_kws={"label": "Proportion"}, ax=ax2
        )
        ax2.set_title("Confusion Matrix (Row-Normalised)", fontsize=13, fontweight="bold")
        ax2.set_ylabel("True Label", fontsize=11)
        ax2.set_xlabel("Predicted Label", fontsize=11)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha="right")

        plt.tight_layout()
        path = self.save_dir / "confusion_matrix.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved → {path}")
        return cm, cm_norm

    # ── Metrics ───────────────────────────────────────────────────────

    def compute_metrics(self, cm: np.ndarray) -> dict:
        """Compute per-class and global metrics for all 5 classes."""
        metrics = {
            "overall_accuracy": accuracy_score(self.all_targets, self.all_preds)
        }

        ious, f1s, precisions, recalls = [], [], [], []

        for cls in range(NUM_CLASSES):
            tp = cm[cls, cls]
            fp = cm[:, cls].sum() - tp
            fn = cm[cls, :].sum() - tp

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
            iou  = tp / (tp+fp+fn)         if (tp+fp+fn) > 0 else 0.0

            name = LABEL_NAMES[cls]
            metrics[f"{name}/precision"] = prec
            metrics[f"{name}/recall"]    = rec
            metrics[f"{name}/f1"]        = f1
            metrics[f"{name}/iou"]       = iou

            ious.append(iou)
            f1s.append(f1)
            precisions.append(prec)
            recalls.append(rec)

        metrics["mean_iou"]       = float(np.mean(ious))
        metrics["mean_f1"]        = float(np.mean(f1s))
        metrics["mean_precision"] = float(np.mean(precisions))
        metrics["mean_recall"]    = float(np.mean(recalls))

        return metrics

    # ── Per-Class IoU Bar Chart ───────────────────────────────────────

    def plot_iou_bar(self, metrics: dict):
        """Save a per-class IoU bar chart."""
        names  = [LABEL_NAMES[i] for i in range(NUM_CLASSES)]
        ious   = [metrics[f"{n}/iou"] for n in names]
        colors = [PALETTE[i] for i in range(NUM_CLASSES)]

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(names, ious, color=colors, edgecolor="black", linewidth=0.6)

        for bar, val in zip(bars, ious):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10
            )

        ax.axhline(
            metrics["mean_iou"], color="black", linestyle="--",
            linewidth=1.2, label=f"Mean IoU = {metrics['mean_iou']:.3f}"
        )
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("IoU", fontsize=12)
        ax.set_title("Per-Class IoU — Test Set", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        path = self.save_dir / "iou_per_class.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved → {path}")

    # ── Sample Visualisation ──────────────────────────────────────────

    def visualize_samples(self):
        """Save grid of pre/post/ground-truth/prediction for randomly selected samples."""
        num = len(self.samples)
        print(f"\n[Evaluation] Visualising {num} randomly selected samples...")

        fig, axes = plt.subplots(
            num, 4,
            figsize=(20, 5 * num),
            gridspec_kw={"width_ratios": [1, 1, 1, 1]}
        )
        if num == 1:
            axes = axes.reshape(1, -1)

        col_titles = ["Pre-Disaster", "Post-Disaster", "Ground Truth", "Prediction"]
        for col, title in enumerate(col_titles):
            axes[0, col].set_title(title, fontsize=12, fontweight="bold", pad=8)

        for idx, s in enumerate(self.samples):
            pre_rgb    = np.clip(s["pre"],  0, 1)
            post_rgb   = np.clip(s["post"], 0, 1)
            target_rgb = labels_to_rgb(s["target"])
            pred_rgb   = labels_to_rgb(s["pred"])

            valid  = s["target"] != IGNORE_INDEX
            px_acc = (s["pred"][valid] == s["target"][valid]).mean() * 100

            axes[idx, 0].imshow(pre_rgb);    axes[idx, 0].axis("off")
            axes[idx, 1].imshow(post_rgb);   axes[idx, 1].axis("off")
            axes[idx, 2].imshow(target_rgb); axes[idx, 2].axis("off")
            axes[idx, 3].imshow(pred_rgb);   axes[idx, 3].axis("off")

            # Stem label on left, accuracy on right
            axes[idx, 0].set_ylabel(
                s["stem"], fontsize=7, rotation=90, labelpad=4
            )
            axes[idx, 3].set_xlabel(
                f"Px Acc: {px_acc:.1f}%", fontsize=10, labelpad=3
            )

        # Colour legend
        legend_handles = [
            mpatches.Patch(facecolor=PALETTE[i], edgecolor="black", label=LABEL_NAMES[i])
            for i in range(NUM_CLASSES)
        ]
        fig.legend(
            handles=legend_handles, loc="lower center",
            ncol=NUM_CLASSES, fontsize=10,
            bbox_to_anchor=(0.5, -0.01)
        )

        plt.tight_layout()
        path = self.save_dir / "test_samples.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved → {path}")

    # ── Print + Save ──────────────────────────────────────────────────

    def print_metrics(self, metrics: dict):
        print("\n" + "=" * 60)
        print("EVALUATION METRICS — All 5 Classes")
        print("=" * 60)
        print(f"\n  Overall Accuracy : {metrics['overall_accuracy']:.4f}")
        print(f"  Mean IoU         : {metrics['mean_iou']:.4f}")
        print(f"  Mean F1          : {metrics['mean_f1']:.4f}")
        print(f"  Mean Precision   : {metrics['mean_precision']:.4f}")
        print(f"  Mean Recall      : {metrics['mean_recall']:.4f}")
        print("\n  Per-Class Metrics:")
        print("  " + "-" * 56)
        print(f"  {'Class':<16} {'Prec':>8} {'Rec':>8} {'F1':>8} {'IoU':>8}")
        print("  " + "-" * 56)
        for cls in range(NUM_CLASSES):
            name = LABEL_NAMES[cls]
            print(
                f"  {name:<16}"
                f" {metrics[f'{name}/precision']:>8.4f}"
                f" {metrics[f'{name}/recall']:>8.4f}"
                f" {metrics[f'{name}/f1']:>8.4f}"
                f" {metrics[f'{name}/iou']:>8.4f}"
            )
        print("=" * 60)

    def save_metrics(self, metrics: dict):
        path = self.save_dir / "metrics.txt"
        with open(path, "w") as f:
            f.write("EVALUATION METRICS — All 5 Classes\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Overall Accuracy : {metrics['overall_accuracy']:.4f}\n")
            f.write(f"Mean IoU         : {metrics['mean_iou']:.4f}\n")
            f.write(f"Mean F1          : {metrics['mean_f1']:.4f}\n")
            f.write(f"Mean Precision   : {metrics['mean_precision']:.4f}\n")
            f.write(f"Mean Recall      : {metrics['mean_recall']:.4f}\n")
            f.write("\nPer-Class Metrics:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Class':<16} {'Prec':>8} {'Rec':>8} {'F1':>8} {'IoU':>8}\n")
            f.write("-" * 60 + "\n")
            for cls in range(NUM_CLASSES):
                name = LABEL_NAMES[cls]
                f.write(
                    f"{name:<16}"
                    f" {metrics[f'{name}/precision']:>8.4f}"
                    f" {metrics[f'{name}/recall']:>8.4f}"
                    f" {metrics[f'{name}/f1']:>8.4f}"
                    f" {metrics[f'{name}/iou']:>8.4f}\n"
                )
        print(f"  ✓ Metrics saved → {path}")

    # ── Entry Point ───────────────────────────────────────────────────

    def run(self) -> dict:
        """Run full evaluation pipeline and return metrics dict."""
        self.evaluate()
        cm, _       = self.compute_confusion_matrix()
        metrics     = self.compute_metrics(cm)
        self.plot_iou_bar(metrics)
        self.visualize_samples()
        self.print_metrics(metrics)
        self.save_metrics(metrics)
        return metrics