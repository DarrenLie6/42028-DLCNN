#!/usr/bin/env python3
"""
Evaluate trained Siamese UNet model on test set.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --config configs/train_config.yaml
    python scripts/evaluate.py --checkpoint checkpoints/Unet.pth --output eval_results
"""

import argparse
from pathlib import Path
import yaml
import torch
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.siamese_unet import SiameseUNet
from src.data.dataloader import get_dataloaders
from src.evaluation.evaluate_model import ModelEvaluator


# ── Dot-accessible config ─────────────────────────────────────────────

class DotConfig:
    """Recursively converts a nested dict to dot-accessible attributes."""
    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, DotConfig(v) if isinstance(v, dict) else v)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate Siamese UNet on test set")
    parser.add_argument(
        "--config",
        default="configs/train_config.yaml",
        help="Path to train config (default: configs/train_config.yaml)"
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/Unet.pth",
        help="Path to model checkpoint (default: checkpoints/Unet.pth)"
    )
    parser.add_argument(
        "--output",
        default="eval_results",
        help="Output directory for results (default: eval_results)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of test samples to visualize (default: 5)"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available, else cpu)"
    )

    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)

    cfg = DotConfig(cfg_dict)
    print(f"[INFO] Loaded config from {config_path}")
    print(f"[INFO] Dataset root : {cfg.data.root_dir}")

    # ── Setup device ──────────────────────────────────────────────────
    device = torch.device(args.device)
    print(f"[INFO] Using device : {device}")

    # ── Load model ────────────────────────────────────────────────────
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"[INFO] Loading model from {checkpoint_path}")

    model      = SiameseUNet(num_classes=5, pretrained=False)
    checkpoint = torch.load(
        checkpoint_path,
        map_location = device,
        weights_only = False,
    )

    # Extract model weights from checkpoint wrapper
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device).eval()

    print(f"[INFO] Model loaded  — epoch {checkpoint['epoch']} | "
          f"val mIoU = {checkpoint['val_mean_iou']:.4f}")

    # ── Get test dataloader ───────────────────────────────────────────
    print(f"\n[INFO] Loading test data...")
    _, _, test_loader = get_dataloaders(cfg)
    print(f"[INFO] Test set size : {len(test_loader.dataset)} tiles")

    # ── Run evaluation ────────────────────────────────────────────────
    print(f"\n[INFO] Starting evaluation...")
    evaluator = ModelEvaluator(
        model       = model,
        test_loader = test_loader,
        device      = device,
        save_dir    = args.output,
    )

    metrics = evaluator.run()

    print(f"\n[INFO] Evaluation complete!")
    print(f"[INFO] Results saved to : {args.output}/")
    print(f"\n  Mean IoU : {metrics['mean_iou']:.4f}")
    print(f"  Mean F1  : {metrics['mean_f1']:.4f}")
    print(f"  Accuracy : {metrics['overall_accuracy']:.4f}")


if __name__ == "__main__":
    main()