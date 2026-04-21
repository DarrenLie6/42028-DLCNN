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
        
        for epoch in range(1, self.num_epochs + 1):
            epoch_start = time.time()

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
                self.best_mean_iou = val_mean_iou #saves the best weights
                self.epochs_no_improve = 0
                self._save_checkpoint(epoch, val_mean_iou)
            else:
                self.epochs_no_improve += 1

            # record history
            record = {"epoch": epoch, "lr": current_lr, **train_stats, **val_stats}
            self.history.append(record)

            # early stopping
            if self.epochs_no_improve >= self.patience:
                print(
                    f"\n EARLY STOPPING triggered after {epoch} epochs "
                    f"({self.patience} epochs without improvement)."
                )
                break

        print(f"\n  Training complete. Best val mean_iou: {self.best_mean_iou:.4f}")
        return self.history
        
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
            "train/loss": total_loss / n_batches,
            "train/ce_loss": total_ce   / n_batches,
            "train/dice_loss": total_dice / n_batches,
            **{f"train/{k}": v for k, v in metrics.items()},
            # Expose mean_iou at top level for early stopping / checkpointing
            "mean_iou": metrics["mean_iou"],
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
            "val/loss": total_loss / n_batches,
            "val/ce_loss": total_ce   / n_batches,
            "val/dice_loss": total_dice / n_batches,
            **{f"val/{k}": v for k, v in metrics.items()},
            # Expose val mean_iou at top level for checkpoint gating
            "mean_iou": metrics["mean_iou"],
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
        
    def load_checkpoint(self, path: str) -> int:
        """
        Restore trainer state from a saved checkpoint.
        """
        
        ckpt = torch.load(path, map_location=self.device)
        
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
