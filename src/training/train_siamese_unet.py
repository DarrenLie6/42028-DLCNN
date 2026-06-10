"""
Training script for the Siamese Attention U-Net (ResNet-34) on xView2.

Bi-temporal (pre + post) input, focal-enabled loss. Reuses the existing
`Trainer` — the model takes a 6-channel image, so nothing in the loop changes.

Usage:
    python -m src.training.train_siamese_unet --config configs/siamese_unet_config.yaml
"""

from __future__ import annotations
import argparse
import os
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


def _repair_ssl_cert_file() -> None:
    """Ensure SSL_CERT_FILE points to a valid CA bundle.

    conda's openssl activation script may leave SSL_CERT_FILE pointing at a
    path that doesn't exist, which makes httpx (used by huggingface_hub to
    download timm pretrained weights) crash with FileNotFoundError. If the
    var is unset or stale, fall back to certifi's bundled CA file.
    """
    current = os.environ.get("SSL_CERT_FILE")
    if current and os.path.exists(current):
        return
    try:
        import certifi
    except ImportError:
        os.environ.pop("SSL_CERT_FILE", None)
        return
    os.environ["SSL_CERT_FILE"] = certifi.where()


_repair_ssl_cert_file()

from src.data.siamese_xview2_dataset import (
    SiameseXViewDataset,
    build_siamese_train_aug,
    build_siamese_val_aug,
)
from src.models.siamese_attention_unet import build_siamese_attention_unet
from src.training.trainer import Trainer
from src.training.losses import CombinedLoss


def main(config_path: str, resume: str | None = None):
    cfg = OmegaConf.load(config_path)
    print("[Config]")
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] Using: {device}")

    # ---- Data ------------------------------------------------------------
    print("\n[Data] Building bi-temporal datasets...")
    train_ds = SiameseXViewDataset(
        root_dir=cfg.data.root_dir, cfg=cfg, mode="train",
        transform=build_siamese_train_aug(cfg),
    )
    val_ds = SiameseXViewDataset(
        root_dir=cfg.data.root_dir, cfg=cfg, mode="val",
        transform=build_siamese_val_aug(cfg),
    )
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size,
        shuffle=True, num_workers=cfg.training.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size,
        shuffle=False, num_workers=cfg.training.num_workers,
    )

    # ---- Model -----------------------------------------------------------
    print("\n[Model] Building Siamese Attention U-Net...")
    model = build_siamese_attention_unet(
        num_classes=cfg.data.num_classes,
        backbone=cfg.model.get("backbone", "resnet34"),
        pretrained=cfg.model.get("pretrained", True),
        dropout_p=cfg.model.get("dropout_p", 0.15),
        freeze_encoder_bn=cfg.model.get("freeze_encoder_bn", False),
        device=device,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")

    # ---- Loss (focal enabled) -------------------------------------------
    lc = cfg.get("loss", {})
    criterion = CombinedLoss(
        num_classes=cfg.data.num_classes,
        ignore_index=-100,
        class_weights=list(lc.get("class_weights", [0.5, 5.0, 7.0, 10.0])),
        ce_weight=lc.get("ce_weight", 0.4),
        dice_weight=lc.get("dice_weight", 0.3),
        focal_weight=lc.get("focal_weight", 0.3),
        use_focal=True,
    )
    print(f"  Loss: CombinedLoss(CE+Dice+Focal) weights "
          f"({criterion.ce_w}/{criterion.dice_w}/{criterion.focal_w})")

    # ---- Trainer ---------------------------------------------------------
    print("\n[Trainer] Initializing...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        num_epochs=cfg.training.epochs,
        patience=cfg.training.patience,
        checkpoint_dir=cfg.training.checkpoint_dir,
        t_max=cfg.training.epochs,
        eta_min=cfg.training.min_lr,
        criterion=criterion,
        use_amp=cfg.training.get("use_amp", True),
        amp_init_scale=float(cfg.training.get("amp_init_scale", 2 ** 13)),
    )

    # ---- Resume (optional) ----------------------------------------------
    start_epoch = 0
    if resume:
        resume_path = resume
        if resume_path == "auto":
            resume_path = os.path.join(cfg.training.checkpoint_dir, "UNet.pth")
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"--resume given but no checkpoint at {resume_path}")
        print(f"\n[Resume] Loading checkpoint: {resume_path}")
        start_epoch = trainer.load_checkpoint(resume_path)
        print(f"[Resume] Continuing from epoch {start_epoch + 1} "
              f"to {cfg.training.epochs} (best mIoU={trainer.best_mean_iou:.4f})")

    print("\n[Training] Starting...")
    history = trainer.fit(start_epoch=start_epoch)

    print("\n✓ Training complete!")
    print(f"  Best val mean IoU: {trainer.best_mean_iou:.4f}")
    print(f"  Checkpoints: {cfg.training.checkpoint_dir}")
    return trainer, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Siamese Attention U-Net on xView2")
    parser.add_argument(
        "--config", type=str, default="configs/siamese_unet_config.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--resume", type=str, nargs="?", const="auto", default=None,
        help="Resume training from a checkpoint. Pass a path, or use the flag "
             "with no value to auto-load <checkpoint_dir>/UNet.pth.",
    )
    args = parser.parse_args()
    main(args.config, resume=args.resume)
