import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from src.training.metrics import SegmentationMetrics
from src.training.losses  import CombinedLoss

"""
Training and validation loop for BRIGHT Siamese UNet.

  - AdamW optimizer
  - CosineAnnealingLR scheduler
  - FP16 mixed precision (torch.amp)
  - Train / val loop with SegmentationMetrics
  - Best checkpoint saving gated on mean_iou
  - Early stopping
"""

NUM_CLASSES   = 4
IGNORE_INDEX  = 0
LABEL_NAMES   = {0: "Background", 1: "Intact", 2: "Damaged", 3: "Destroyed"}
CLASS_WEIGHTS = [0.0, 1.0, 7.8, 13.0]

class Trainer:
    """Encapsulate the full training loop for the Siamese UNet"""
    
    def __init__(self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        num_epochs: int   = 50,
        patience: int   = 10,
        checkpoint_dir: str   = "checkpoints",
        t_max: int   = 50,
        eta_min: float = 1e-6,
    ):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # scheduler: cosine decay from lr 
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=t_max,
            eta_min=eta_min
        )
        
        # loss
        self.criterion = CombinedLoss(
            class_weights=CLASS_WEIGHTS,
            num_classes=NUM_CLASSES,
            ignore_index=IGNORE_INDEX
        ).to(device)
        
        # FP16 scaler
        self.scaler = GradScaler(enabled=device.type == "cuda")
        
        # metrics
        self.train_metrics = SegmentationMetrics(device=device)
        self.val_metrics = SegmentationMetrics(device=device)
        
        # state
        self.best_mean_iou = 0.0
        self.epochs_no_improve = 0
        self.history = []   # list of per-epoch dicts
        
    # public entry point
    def fit(self) -> list[dict]:
        """Run the full training loop."""

        # Freeze encoder for first 10 epochs
        self._freeze_encoder()

        for epoch in range(1, self.num_epochs + 1):
            epoch_start = time.time()

            # Unfreeze encoder at epoch 10 and halve LR
            if epoch == 10:
                self._unfreeze_encoder()
                for g in self.optimizer.param_groups:
                    g['lr'] = g['lr'] * 0.5
                print(f" Encoder unfrozen at epoch {epoch} | LR halved to {self.optimizer.param_groups[0]['lr']:.2e}")

            train_stats = self._train_epoch(epoch)
            val_stats   = self._val_epoch(epoch)

            # Step scheduler after each epoch
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # log
            elapsed = time.time() - epoch_start
            self._log_epoch(epoch, train_stats, val_stats, current_lr, elapsed)

            # checkpoint
            val_mean_iou = val_stats["mean_iou"]
            if val_mean_iou > self.best_mean_iou:
                self.best_mean_iou = val_mean_iou
                self.epochs_no_improve = 0
                self._save_checkpoint(epoch, val_mean_iou)
            else:
                self.epochs_no_improve += 1

            # record history
            record = {"epoch": epoch, "lr": current_lr, **train_stats, **val_stats}
            self.history.append(record)

            # early stopping — only apply after encoder is unfrozen
            if epoch > 10 and self.epochs_no_improve >= self.patience:
                print(
                    f"\n EARLY STOPPING triggered after {epoch} epochs "
                    f"({self.patience} epochs without improvement)."
                )
                break

        print(f"\n  Training complete. Best val mean_iou: {self.best_mean_iou:.4f}")
        return self.history
    
    def plot_history(self, save_dir: str = "checkpoints") -> None:
        """
        Saves training curves and confusion matrix after training completes.
            - training_curves.png  — loss + mIoU + mAcc per epoch
            - confusion_matrix.png — normalised CM from best val epoch
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        os.makedirs(save_dir, exist_ok=True)
        epochs = [r["epoch"] for r in self.history]

        # ── Figure 1: Training Curves ──────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Training Curves — BRIGHT Siamese UNet", fontsize=14)

        # Loss
        axes[0].plot(epochs, [r["train/loss"] for r in self.history], label="Train")
        axes[0].plot(epochs, [r["val/loss"]   for r in self.history], label="Val")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True)

        # mIoU
        axes[1].plot(epochs, [r["train/mean_iou"] for r in self.history], label="Train")
        axes[1].plot(epochs, [r["val/mean_iou"]   for r in self.history], label="Val")
        axes[1].set_title("Mean IoU (classes 1–3)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("mIoU")
        axes[1].legend()
        axes[1].grid(True)

        # Mean Accuracy
        axes[2].plot(epochs, [r.get("train/mean_acc", 0) for r in self.history], label="Train")
        axes[2].plot(epochs, [r.get("val/mean_acc",   0) for r in self.history], label="Val")
        axes[2].set_title("Mean Accuracy (classes 1–3)")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Accuracy")
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        curve_path = os.path.join(save_dir, "training_curves.png")
        plt.savefig(curve_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f" Training curves saved → {curve_path}")

        # reload Best Checkpoint for CM 
        best_path = os.path.join(self.checkpoint_dir, "UNet.pth")
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(ckpt["model_state"])
            print(f" Reloaded best weights for CM (val mIoU={ckpt['val_mean_iou']:.4f})")

            # Rebuild val_metrics from best weights
            self.val_metrics.reset()
            self.model.eval()
            with torch.no_grad():
                for batch in self.val_loader:
                    optical       = batch["optical"].to(self.device)
                    sar           = batch["sar"].to(self.device)
                    targets       = batch["label"].to(self.device)
                    optical_valid = batch.get("optical_valid", None)
                    if optical_valid is not None:
                        optical_valid = optical_valid.to(self.device)
                    with autocast(
                        device_type=self.device.type,
                        enabled=self.device.type == "cuda"
                    ):
                        logits = self.model(optical, sar, optical_valid)
                    self.val_metrics.update(logits, targets)
            print(f" Val metrics rebuilt from best checkpoint")
        else:
            print(f" Warning: No checkpoint found at {best_path} — CM will use last epoch")

        # Figure 2: Confusion Matrix 
        cm = self.val_metrics.conf_matrix.cpu().numpy().astype(float)

        # Row-normalise so each cell = recall per class
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax)

        class_names = [LABEL_NAMES[i] for i in range(NUM_CLASSES)]
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(
            f"Confusion Matrix (row-normalised) — Best Val Epoch\n"
            f"(val mIoU={self.best_mean_iou:.4f})"
        )

        # Annotate cells
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                val = cm_norm[i, j]
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=color, fontsize=10)

        plt.tight_layout()
        cm_path = os.path.join(save_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f" Confusion matrix saved → {cm_path}")
        
    # private helper functions
    def _train_epoch(self, epoch: int) -> dict:
        """One full pass over the training set."""
        
        self.model.train()
        self.train_metrics.reset()

        total_loss = 0.0
        total_ce = 0.0
        total_dice = 0.0
        n_batches = 0

        for batch in self.train_loader:
            optical = batch["optical"].to(self.device)      # (B,3,H,W) or None
            sar = batch["sar"].to(self.device)          # (B,1,H,W)
            targets = batch["label"].to(self.device)         # (B,H,W) long
            optical_valid = batch.get("optical_valid", None)

            if optical_valid is not None:
                optical_valid = optical_valid.to(self.device)   # (B,) bool

            self.optimizer.zero_grad()

            # Forward + loss under FP16
            with autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                logits = self.model(optical, sar, optical_valid)  # (B,4,H,W)
                loss, ce_loss, dice_loss = self.criterion(logits, targets)

            # Backward propagation with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate losses (detach before .item() to free graph)
            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_dice += dice_loss.item()
            n_batches += 1

            # Accumulate metrics (no grad needed)
            self.train_metrics.update(logits.detach(), targets)

        metrics = self.train_metrics.compute()

        return {
            "train/loss":      total_loss / n_batches,
            "train/ce_loss":   total_ce   / n_batches,
            "train/dice_loss": total_dice / n_batches,
            "train/mean_iou":  metrics["mean_iou"],      
            "train/mean_f1":   metrics["mean_f1"],        
            "train/mean_acc":  metrics["mean_acc"],      
            **{f"train/{k}": v for k, v in metrics.items()},
            "mean_iou": metrics["mean_iou"],              # top-level for early stopping
        }
            
    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> dict:
        """One full pass over the validation set (no gradients)."""
        
        self.model.eval()
        self.val_metrics.reset()

        total_loss = 0.0
        total_ce  = 0.0
        total_dice = 0.0
        n_batches = 0

        for batch in self.val_loader:
            optical = batch["optical"].to(self.device)
            sar = batch["sar"].to(self.device)
            targets = batch["label"].to(self.device)
            optical_valid = batch.get("optical_valid", None)

            if optical_valid is not None:
                optical_valid = optical_valid.to(self.device)

            with autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                logits = self.model(optical, sar, optical_valid)
                loss, ce_loss, dice_loss = self.criterion(logits, targets)
                
            # ── DEBUG: add these lines temporarily ────────────────
            if torch.isnan(loss):
                print(f"  NaN detected!")
                print(f"  ce_loss   = {ce_loss.item()}")
                print(f"  dice_loss = {dice_loss.item()}")
                print(f"  logits    min={logits.min().item():.4f} max={logits.max().item():.4f} nan={torch.isnan(logits).any()}")
                print(f"  targets   min={targets.min().item()} max={targets.max().item()} nan={torch.isnan(targets.float()).any()}")
                print(f"  optical   min={optical.min().item():.4f} max={optical.max().item():.4f} nan={torch.isnan(optical).any()}")
                print(f"  sar       min={sar.min().item():.4f} max={sar.max().item():.4f} nan={torch.isnan(sar).any()}")
                break
             
            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_dice += dice_loss.item()
            n_batches += 1

            self.val_metrics.update(logits, targets)

        metrics = self.val_metrics.compute()

        return {
            "val/loss":      total_loss / n_batches,
            "val/ce_loss":   total_ce   / n_batches,
            "val/dice_loss": total_dice / n_batches,
            "val/mean_iou":  metrics["mean_iou"],         # ← prefixed
            "val/mean_f1":   metrics["mean_f1"],          # ← prefixed
            "val/mean_acc":  metrics["mean_acc"],         # ← prefixed
            **{f"val/{k}": v for k, v in metrics.items()},
            "mean_iou": metrics["mean_iou"],              # top-level for checkpointing
        }
        
    def _save_checkpoint(self, epoch: int, val_mean_iou: float) -> None:
        """Save model + optimizer + scheduler state to disk."""
        
        path = os.path.join(self.checkpoint_dir, "UNet.pth")
        torch.save(
            {
                "epoch":        epoch,
                "model_state":  self.model.state_dict(),
                "optim_state":  self.optimizer.state_dict(),
                "sched_state":  self.scheduler.state_dict(),
                "scaler_state": self.scaler.state_dict(),
                "val_mean_iou": val_mean_iou,
            },
            path,
        )
        print(f" Checkpoint saved → {path}  (val mean_iou={val_mean_iou:.4f})")
        
        
    def _freeze_encoder(self):
        for name, param in self.model.named_parameters():
            if "encoder" in name:
                param.requires_grad = False
        frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        print(f" Encoder frozen ({frozen/1e6:.1f}M params)")

    def _unfreeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = True
        print(f" Encoder unfrozen")
        
    def load_checkpoint(self, path: str) -> int:
        """
        Restore trainer state from a saved checkpoint.
        """
        
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        self.scheduler.load_state_dict(ckpt["sched_state"])
        self.scaler.load_state_dict(ckpt["scaler_state"])
        self.best_mean_iou = ckpt["val_mean_iou"]
        
        print(f"Loaded checkpoint from {path}  (val mean_iou={self.best_mean_iou:.4f})")
        return ckpt["epoch"]
    
    @staticmethod
    def _log_epoch(
        epoch: int,
        train: dict,
        val: dict,
        lr: float,
        elapsed: float,
    ) -> None:
        
        print(
            f"Epoch {epoch:03d} | "
            f"lr={lr:.2e} | "
            f"train loss={train['train/loss']:.4f} "
            f"mIoU={train['mean_iou']:.4f} | "
            f"val loss={val['val/loss']:.4f} "
            f"mIoU={val['mean_iou']:.4f} | "
            f"{elapsed:.1f}s"
        )
