"""
Training script for the transformer-based DeepLabV3+ on xView2.

This trains the alternative `TransformerDeepLabV3Plus` model (MiT/SegFormer
encoder + DeepLabV3+ decoder) while reusing the same dataset, augmentation and
trainer as the ResNet baseline. The baseline script (`train_deeplabv3.py`) is
left untouched.

Usage:
    python -m src.models.deeplabv3.train_deeplabv3plus_transformer --config configs/deeplabv3plus_transformer_config.yaml
"""

from __future__ import annotations
import argparse
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.models.deeplabv3 import SemanticSegmentationXViewDataset
from src.models.deeplabv3.deeplabv3plus_transformer import build_transformer_semantic_model
from src.models.deeplabv3.deeplabv3_trainer import SemanticSegmentationTrainer
from src.data.augmentation_utils import build_train_aug, build_val_aug
from src.data.siamese_xview2_dataset import (
    SiameseXViewDataset,
    build_siamese_train_aug,
    build_siamese_val_aug,
)


def main(config_path: str, resume: str | None = None):
    cfg = OmegaConf.load(config_path)
    print("[Config]")
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] Using: {device}")

    bitemporal = bool(cfg.model.get("bitemporal", False))

    # ---- Data ------------------------------------------------------------
    # Bi-temporal: load 6-channel [pre|post] tiles with geometry-locked augs so
    # pre and post stay pixel-registered. Single-input: post-only 3-channel.
    print(f"\n[Data] Building datasets (bitemporal={bitemporal})...")
    if bitemporal:
        train_dataset = SiameseXViewDataset(
            root_dir=cfg.data.root_dir, cfg=cfg, mode="train",
            transform=build_siamese_train_aug(cfg),
        )
        val_dataset = SiameseXViewDataset(
            root_dir=cfg.data.root_dir, cfg=cfg, mode="val",
            transform=build_siamese_val_aug(cfg),
        )
    else:
        train_dataset = SemanticSegmentationXViewDataset(
            root_dir=cfg.data.root_dir, cfg=cfg, mode="train",
            transform=build_train_aug(cfg),
        )
        val_dataset = SemanticSegmentationXViewDataset(
            root_dir=cfg.data.root_dir, cfg=cfg, mode="val",
            transform=build_val_aug(cfg),
        )
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.training.batch_size,
        shuffle=True, num_workers=cfg.training.num_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.training.batch_size,
        shuffle=False, num_workers=cfg.training.num_workers,
    )

    # ---- Model -----------------------------------------------------------
    variant = cfg.model.get("variant", "b1")
    pretrained = cfg.model.get("pretrained", True)
    timm_backbone = cfg.model.get("timm_backbone", None)
    print(f"\n[Model] Building Transformer DeepLabV3+ (MiT-{variant}, pretrained={pretrained})...")
    model = build_transformer_semantic_model(
        num_classes=cfg.data.num_classes,
        variant=variant,
        aux_loss=cfg.model.get("aux_loss", True),
        pretrained=pretrained,
        timm_backbone=timm_backbone,
        bitemporal=bitemporal,
        device=device,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")

    # ---- Trainer ---------------------------------------------------------
    # Loss config (focal can be enabled for the extreme class imbalance).
    lc = cfg.get("loss", {})
    print("\n[Trainer] Initializing...")
    trainer = SemanticSegmentationTrainer(
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
        warmup_epochs=cfg.training.get("warmup_epochs", 5),
        class_weights=list(lc["class_weights"]) if lc.get("class_weights") is not None else None,
        ce_weight=lc.get("ce_weight", 0.5),
        dice_weight=lc.get("dice_weight", 0.5),
        focal_weight=lc.get("focal_weight", 0.0),
        use_focal=lc.get("use_focal", False),
    )

    # ---- Resume (optional) ----------------------------------------------
    start_epoch = 0
    if resume:
        start_epoch = trainer.load_checkpoint(resume)

    print("\n[Training] Starting...")
    history = trainer.fit(start_epoch=start_epoch)

    print("\n✓ Training complete!")
    print(f"  Best mean IoU: {trainer.best_mean_iou:.4f}")
    print(f"  Checkpoints: {cfg.training.checkpoint_dir}")
    return trainer, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train transformer DeepLabV3+ on xView2"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/deeplabv3plus_transformer_config.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a latest.pth checkpoint to resume training from "
             "(e.g. checkpoints/semantic_seg_transformer/latest.pth)",
    )
    args = parser.parse_args()
    main(args.config, resume=args.resume)
