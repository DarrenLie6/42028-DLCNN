"""
Training script for semantic segmentation on xView2 dataset.

Usage:
    python -m src.models.deeplabv3.train_deeplabv3 --config configs\deeplabv3_config.yaml
"""

from __future__ import annotations
import argparse
import torch
from pathlib import Path
from omegaconf import OmegaConf

from src.models.deeplabv3 import (
    build_semantic_model,
    SemanticSegmentationXViewDataset,
)
from src.models.deeplabv3.deeplabv3_trainer import SemanticSegmentationTrainer
from src.data.augmentation_utils import build_train_aug, build_val_aug
from torch.utils.data import DataLoader


def main(config_path: str):
    """
    Main training script.
    
    Args:
        config_path: Path to YAML config file
    """
    # Load config
    cfg = OmegaConf.load(config_path)
    print("[Config]")
    print(OmegaConf.to_yaml(cfg))
    
    # Setup device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"\n[Device] Using: {device}")
    
    # Build datasets
    print("\n[Data] Building datasets...")
    
    train_aug = build_train_aug(cfg)
    val_aug = build_val_aug(cfg)
    
    train_dataset = SemanticSegmentationXViewDataset(
        root_dir=cfg.data.root_dir,
        cfg=cfg,
        mode="train",
        transform=train_aug,
    )
    
    val_dataset = SemanticSegmentationXViewDataset(
        root_dir=cfg.data.root_dir,
        cfg=cfg,
        mode="val",
        transform=val_aug,
    )
    
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    
    # Build dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )
    
    # Build model
    print("\n[Model] Building DeepLabV3...")
    model = build_semantic_model(
        num_classes=cfg.data.num_classes,
        pretrained=True,
        device=device,
    )
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Build trainer
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
    )
    
    # Train
    print("\n[Training] Starting...")
    history = trainer.fit(start_epoch=0)
    
    print("\n✓ Training complete!")
    print(f"  Best mean IoU: {trainer.best_mean_iou:.4f}")
    print(f"  Checkpoints: {cfg.training.checkpoint_dir}")
    
    return trainer, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train semantic segmentation on xView2"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/deeplabv3_config.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()
    
    main(args.config)
