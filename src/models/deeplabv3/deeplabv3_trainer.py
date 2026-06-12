"""
Trainer for semantic segmentation on xView2 dataset.
"""

from __future__ import annotations
import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from typing import Dict, List
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for training loops
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.training.losses import CombinedLoss
from src.training.metrics import SegmentationMetrics


NUM_CLASSES = 4
IGNORE_INDEX = -100
LABEL_NAMES = {0: "Background", 1: "Intact", 2: "Damaged", 3: "Destroyed"}
CLASS_WEIGHTS = [0.5, 5.0, 9.0, 15.0]


class SemanticSegmentationTrainer:
    """
    Trainer for semantic segmentation on xView2.

    Features:
      - Mixed precision training (FP16)
      - Gradient clipping
      - Early stopping
      - Checkpoint saving
      - Live training curves (loss + mIoU), saved after every epoch
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        num_epochs: int = 50,
        patience: int = 10,
        checkpoint_dir: str = "checkpoints/semantic_seg",
        t_max: int = 50,
        eta_min: float = 1e-6,
        warmup_epochs: int = 5,
        class_weights: list | None = None,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        focal_weight: float = 0.0,
        use_focal: bool = False,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        # ------------------------------------------------------------------
        # Scheduler — linear warm-up → cosine annealing.
        #
        # The previous implementation used a bare CosineAnnealingLR that
        # started at the full LR from step 0. With a randomly-initialised
        # segmentation head on top of a pretrained backbone (and *especially*
        # a transformer encoder) this produces large, noisy gradients in the
        # first epochs that corrupt the pretrained features before the head
        # has stabilised. We now warm the LR up linearly for `warmup_epochs`,
        # then cosine-anneal over the REMAINING epochs so the schedule still
        # reaches `eta_min` exactly at the final epoch.
        # ------------------------------------------------------------------
        self.warmup_epochs = max(0, int(warmup_epochs))
        if self.warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1e-3,            # start at 0.1% of base LR
                end_factor=1.0,
                total_iters=self.warmup_epochs,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, t_max - self.warmup_epochs),   # account for warm-up
                eta_min=eta_min,
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[self.warmup_epochs],
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=t_max, eta_min=eta_min
            )

        # Loss — configurable so focal can be enabled for the extreme class
        # imbalance (Background ~86% vs Destroyed ~0.9%). When use_focal is
        # True the CombinedLoss returns a 4-tuple; _extract_loss handles both.
        self.criterion = CombinedLoss(
            class_weights=class_weights if class_weights is not None else CLASS_WEIGHTS,
            num_classes=NUM_CLASSES,
            ignore_index=IGNORE_INDEX,
            ce_weight=ce_weight,
            dice_weight=dice_weight,
            focal_weight=focal_weight,
            use_focal=use_focal,
        ).to(device)
        print(
            f"[loss] CombinedLoss(ce={ce_weight}, dice={dice_weight}, "
            f"focal={focal_weight}, use_focal={use_focal}) "
            f"weights={class_weights if class_weights is not None else CLASS_WEIGHTS}"
        )

        # FP16 scaler
        self.scaler = GradScaler(enabled=device.type == "cuda")

        # Metrics
        self.train_metrics = SegmentationMetrics(
            num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX, device=device
        )
        self.val_metrics = SegmentationMetrics(
            num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX, device=device
        )

        # State
        self.best_mean_iou = 0.0
        self.epochs_no_improve = 0
        self.history = []

        # Curve data — populated after each epoch
        self._curve = {
            "epochs":      [],
            "train_loss":  [],
            "val_loss":    [],
            "train_miou":  [],
            "val_miou":    [],
        }

        # Paths for curve outputs
        self._curve_png  = os.path.join(checkpoint_dir, "training_curves.png")
        self._curve_json = os.path.join(checkpoint_dir, "training_history.json")

    # ------------------------------------------------------------------
    # Helper: safely extract scalar tensor from CombinedLoss output
    # CombinedLoss may return a tuple (total, ce, dice) for logging.
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_loss(loss_output) -> torch.Tensor:
        if isinstance(loss_output, (tuple, list)):
            return loss_output[0]
        return loss_output

    # ------------------------------------------------------------------
    # Training curve — called after every epoch
    # ------------------------------------------------------------------
    def _update_curves(self, epoch: int, train_stats: Dict, val_stats: Dict) -> None:
        """Append epoch metrics and redraw the training curve PNG."""
        self._curve["epochs"].append(epoch)
        self._curve["train_loss"].append(train_stats["loss"])
        self._curve["val_loss"].append(val_stats["loss"])
        self._curve["train_miou"].append(train_stats.get("mean_iou", 0.0))
        self._curve["val_miou"].append(val_stats.get("mean_iou", 0.0))

        epochs = self._curve["epochs"]

        fig = plt.figure(figsize=(14, 5), facecolor="#0f0f0f")
        gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

        ax_loss = fig.add_subplot(gs[0])
        ax_miou = fig.add_subplot(gs[1])

        for ax in (ax_loss, ax_miou):
            ax.set_facecolor("#1a1a1a")
            ax.tick_params(colors="#cccccc", labelsize=9)
            ax.xaxis.label.set_color("#cccccc")
            ax.yaxis.label.set_color("#cccccc")
            ax.title.set_color("#ffffff")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333333")
            ax.grid(True, color="#2a2a2a", linewidth=0.7, linestyle="--")

        # ── Loss plot ──────────────────────────────────────────────
        ax_loss.plot(
            epochs, self._curve["train_loss"],
            color="#4f98a3", linewidth=2.0, marker="o", markersize=3,
            label="Train Loss",
        )
        ax_loss.plot(
            epochs, self._curve["val_loss"],
            color="#e8af34", linewidth=2.0, marker="o", markersize=3,
            label="Val Loss",
        )
        # Mark best val loss
        best_loss_epoch = epochs[self._curve["val_loss"].index(min(self._curve["val_loss"]))]
        best_loss_val   = min(self._curve["val_loss"])
        ax_loss.axvline(best_loss_epoch, color="#e8af34", linewidth=0.8, linestyle=":", alpha=0.6)
        ax_loss.annotate(
            f"best {best_loss_val:.4f}",
            xy=(best_loss_epoch, best_loss_val),
            xytext=(8, 8), textcoords="offset points",
            color="#e8af34", fontsize=7.5,
        )
        ax_loss.set_title("Loss", fontsize=12, fontweight="bold", pad=10)
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend(
            framealpha=0.15, edgecolor="#444444",
            labelcolor="#cccccc", fontsize=9,
        )

        # ── mIoU plot ──────────────────────────────────────────────
        ax_miou.plot(
            epochs, self._curve["train_miou"],
            color="#4f98a3", linewidth=2.0, marker="o", markersize=3,
            label="Train mIoU",
        )
        ax_miou.plot(
            epochs, self._curve["val_miou"],
            color="#e8af34", linewidth=2.0, marker="o", markersize=3,
            label="Val mIoU",
        )
        # Mark best val mIoU
        best_miou_epoch = epochs[self._curve["val_miou"].index(max(self._curve["val_miou"]))]
        best_miou_val   = max(self._curve["val_miou"])
        ax_miou.axvline(best_miou_epoch, color="#e8af34", linewidth=0.8, linestyle=":", alpha=0.6)
        ax_miou.annotate(
            f"best {best_miou_val:.4f}",
            xy=(best_miou_epoch, best_miou_val),
            xytext=(8, -14), textcoords="offset points",
            color="#e8af34", fontsize=7.5,
        )
        ax_miou.set_title("Mean IoU", fontsize=12, fontweight="bold", pad=10)
        ax_miou.set_xlabel("Epoch")
        ax_miou.set_ylabel("mIoU")
        ax_miou.set_ylim(0.0, 1.0)
        ax_miou.legend(
            framealpha=0.15, edgecolor="#444444",
            labelcolor="#cccccc", fontsize=9,
        )

        fig.suptitle(
            f"Training Curves — Epoch {epoch} / {self.num_epochs}   "
            f"| Best Val mIoU: {self.best_mean_iou:.4f}",
            color="#ffffff", fontsize=11, y=1.02,
        )

        plt.tight_layout()
        fig.savefig(self._curve_png, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        # Also persist history as JSON for later analysis
        with open(self._curve_json, "w") as f:
            json.dump(self._curve, f, indent=2)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def fit(self, start_epoch: int = 0) -> List[Dict]:
        """Run full training loop."""

        for epoch in range(start_epoch + 1, self.num_epochs + 1):
            epoch_start = time.time()

            train_stats = self._train_epoch(epoch)
            val_stats   = self._val_epoch(epoch)

            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            elapsed = time.time() - epoch_start
            self._log_epoch(epoch, train_stats, val_stats, current_lr, elapsed)

            # Checkpoint on best mean_iou
            val_mean_iou = val_stats.get("mean_iou", 0.0)
            if val_mean_iou > self.best_mean_iou:
                self.best_mean_iou = val_mean_iou
                self.epochs_no_improve = 0
                self._save_checkpoint(epoch, val_mean_iou)
            else:
                self.epochs_no_improve += 1

            record = {"epoch": epoch, "lr": current_lr, **train_stats, **val_stats}
            self.history.append(record)

            # Update and save training curves after every epoch
            self._update_curves(epoch, train_stats, val_stats)
            print(f"  📈 Curves saved → {self._curve_png}")

            # Full-state checkpoint every epoch so training can resume exactly
            # where it stopped (model + optimizer + scheduler + scaler + curves).
            self._save_full_state(epoch)

            # Early stopping
            if epoch > 10 and self.epochs_no_improve >= self.patience:
                print(
                    f"\nEARLY STOPPING triggered after {epoch} epochs "
                    f"({self.patience} epochs without improvement)."
                )
                break

        print(f"\nTraining complete. Best val mean_iou: {self.best_mean_iou:.4f}")
        return self.history

    # ------------------------------------------------------------------
    # Epoch loops
    # ------------------------------------------------------------------
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        self.train_metrics.reset()

        total_loss = 0.0
        count = 0

        pbar = tqdm(self.train_loader, desc=f"Train {epoch}", leave=False)

        for batch in pbar:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # Add after labels = batch["label"].to(self.device)
            if count == 0 and epoch == 1:
                unique, counts = torch.unique(labels, return_counts=True)
                print(f"\n[DEBUG] Label distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

            self.optimizer.zero_grad()

            with autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                outputs = self.model(images)
                logits  = outputs["out"]
                loss    = self._extract_loss(self.criterion(logits, labels))

                if "aux" in outputs:
                    aux_loss = self._extract_loss(self.criterion(outputs["aux"], labels))
                    loss = loss + 0.4 * aux_loss

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.train_metrics.update(logits.detach(), labels)
            total_loss += loss.item()
            count += 1
            pbar.set_postfix({"loss": total_loss / count})

        metrics = self.train_metrics.compute()
        metrics["loss"] = total_loss / max(count, 1)
        return metrics

    def _val_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one validation epoch."""
        self.model.eval()
        self.val_metrics.reset()

        total_loss = 0.0
        count = 0

        pbar = tqdm(self.val_loader, desc=f"Val {epoch}", leave=False)

        with torch.no_grad():
            for batch in pbar:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                with autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                    outputs = self.model(images)
                    logits  = outputs["out"]
                    loss    = self._extract_loss(self.criterion(logits, labels))

                self.val_metrics.update(logits.detach(), labels)
                total_loss += loss.item()
                count += 1
                pbar.set_postfix({"loss": total_loss / count})

        metrics = self.val_metrics.compute()
        metrics["loss"] = total_loss / max(count, 1)
        return metrics

    # ------------------------------------------------------------------
    # Logging & checkpointing
    # ------------------------------------------------------------------
    def _log_epoch(
        self,
        epoch: int,
        train_stats: Dict[str, float],
        val_stats: Dict[str, float],
        current_lr: float,
        elapsed: float,
    ) -> None:
        print(
            f"[Epoch {epoch}] "
            f"train_loss={train_stats['loss']:.4f} | "
            f"val_loss={val_stats['loss']:.4f} | "
            f"val_mIoU={val_stats['mean_iou']:.4f} | "
            f"val_mF1={val_stats['mean_f1']:.4f} | "
            f"lr={current_lr:.2e} | "
            f"time={elapsed:.1f}s"
        )

    def _save_checkpoint(self, epoch: int, mean_iou: float) -> None:
        ckpt_path = os.path.join(
            self.checkpoint_dir,
            f"semantic_seg_best_mIoU_{mean_iou:.4f}_epoch_{epoch}.pth",
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "mean_iou": mean_iou,
            },
            ckpt_path,
        )
        print(f"  ✓ Checkpoint saved: {ckpt_path}")

    # ------------------------------------------------------------------
    # Resume support — full-state checkpoint written every epoch
    # ------------------------------------------------------------------
    def _save_full_state(self, epoch: int) -> None:
        """
        Write a single rolling 'latest.pth' with everything needed to resume
        training byte-for-byte: weights, optimizer, scheduler, AMP scaler,
        early-stopping counters and the accumulated training-curve history.
        Overwrites each epoch so it always reflects the most recent epoch.
        """
        path = os.path.join(self.checkpoint_dir, "latest.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "best_mean_iou": self.best_mean_iou,
                "epochs_no_improve": self.epochs_no_improve,
                "curve": self._curve,
                "history": self.history,
            },
            path,
        )
        print(f"  💾 Full state saved → {path}  (resume from epoch {epoch})")

    def load_checkpoint(self, path: str) -> int:
        """
        Restore full trainer state from a 'latest.pth' (or compatible)
        checkpoint and return the epoch to resume *after*.

        Pass the returned value as `fit(start_epoch=...)`; training then
        continues at epoch+1 and the training-curve PNG extends the existing
        history instead of starting over.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ckpt.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        # Restore early-stopping + best-metric state (fall back gracefully for
        # older best-only checkpoints that only store 'mean_iou').
        self.best_mean_iou = ckpt.get("best_mean_iou", ckpt.get("mean_iou", 0.0))
        self.epochs_no_improve = ckpt.get("epochs_no_improve", 0)

        # Restore curve history so the PNG/JSON continue unbroken.
        if "curve" in ckpt and isinstance(ckpt["curve"], dict):
            self._curve = ckpt["curve"]
        if "history" in ckpt and isinstance(ckpt["history"], list):
            self.history = ckpt["history"]

        epoch = int(ckpt.get("epoch", 0))
        print(
            f"  ↻ Resumed from {path} | epoch={epoch} | "
            f"best_mean_iou={self.best_mean_iou:.4f} | "
            f"curve points restored={len(self._curve.get('epochs', []))}"
        )
        return epoch