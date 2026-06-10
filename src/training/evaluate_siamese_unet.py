"""
Evaluation script for the Siamese Attention U-Net (ResNet-34) on xView2.

This is a *separate* evaluator from the single-image `evaluate.py` and the
DeepLabV3+ evaluators, because the Siamese model has a different input/output
contract:

  * Input  : 6-channel `image` = [pre(3) | post(3)]  (not post-only 3ch)
  * Output : a plain logits tensor (B, num_classes, H, W)  (not {"out": ...})

It reuses the DeepLabV3+ baseline's model-agnostic reporting/plotting helpers
(`save_results`, confusion matrix, per-class bars, sample comparisons) — those
only consume a list of per-sample dicts, so we feed them the *post-disaster*
channels for visualisation.

Outputs (saved to --output-dir):
  - metrics.json              : per-class + mean IoU / F1 / Accuracy
  - confusion_matrix.png      : 4x4 GT-vs-prediction heatmap
  - per_class_metrics.png     : IoU / F1 / Accuracy bar charts
  - sample_comparisons/       : post image | GT overlay | prediction overlay
  - predictions_summary.txt   : per-sample pixel-agreement table

Usage:
    python -m src.training.evaluate_siamese_unet 
        --checkpoint checkpoints/siamese_unet/UNet.pth 
        --config configs/siamese_unet_config.yaml 
        --split test 
        --output-dir evaluation_results/siamese_unet
"""

from __future__ import annotations
import argparse
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.siamese_xview2_dataset import (
    SiameseXViewDataset,
    build_siamese_val_aug,
)
from src.models.siamese_attention_unet import build_siamese_attention_unet
from src.training.metrics import SegmentationMetrics

# Reuse the baseline's model-agnostic reporting / plotting pipeline.
from src.models.deeplabv3.evaluate_semantic_seg import (
    save_results,
    NUM_CLASSES,
    IGNORE_INDEX,
    LABEL_NAMES,
)


def load_siamese_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    cfg,
) -> torch.nn.Module:
    """Rebuild the Siamese Attention U-Net and load trained weights."""
    print(f"[Loading] {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    model = build_siamese_attention_unet(
        num_classes=cfg.data.num_classes,
        backbone=cfg.model.get("backbone", "resnet34"),
        pretrained=False,  # weights come from the checkpoint, not ImageNet
        dropout_p=cfg.model.get("dropout_p", 0.15),
        freeze_encoder_bn=cfg.model.get("freeze_encoder_bn", False),
        device=device,
    )

    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing:
        print(f"  [WARN] Missing keys ({len(missing)}):    {missing[:6]}")
    if unexpected:
        print(f"  [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:6]}")

    model.eval()
    print(f"  Epoch: {ckpt.get('epoch', '?')} | "
          f"Train-time best val mIoU: {ckpt.get('val_mean_iou', '?')}")
    return model


def evaluate_siamese(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    save_dir: str,
) -> dict:
    """Run inference over the loader and save metrics + visual samples."""
    model.eval()
    amp_enabled = device.type == "cuda"
    metrics = SegmentationMetrics(
        num_classes=NUM_CLASSES,
        ignore_index=IGNORE_INDEX,
        device=device,
    )

    all_predictions = []

    print("\n[Evaluation] Running inference...")
    with torch.no_grad():
        for batch in tqdm(data_loader):
            images = batch["image"].to(device)   # (B, 6, H, W) = [pre | post]
            labels = batch["label"].to(device)   # (B, H, W)
            stems  = batch["stem"]

            with autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)           # plain tensor (B, C, H, W)

            metrics.update(logits, labels)
            preds = logits.argmax(dim=1)         # (B, H, W)

            for i, stem in enumerate(stems):
                # Visualise the POST-disaster image (channels 3:6), in [0,1].
                post_np = images[i, 3:].cpu().numpy().transpose(1, 2, 0)
                post_np = np.clip(post_np, 0.0, 1.0)

                all_predictions.append({
                    "stem":  stem,
                    "image": post_np,
                    "pred":  preds[i].cpu().numpy(),
                    "gt":    labels[i].cpu().numpy(),
                    "logits": logits[i].float().cpu().numpy(),
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
    print(f"{'Mean':<15} {results['mean_iou']:>10.4f} "
          f"{results['mean_f1']:>10.4f} {results['mean_acc']:>10.4f}")
    print("-" * 60)

    save_results(results, all_predictions, save_dir)
    return results


def main(
    checkpoint_path: str,
    config_path: str,
    split: str = "test",
    output_dir: str = "evaluation_results/siamese_unet",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")

    cfg   = OmegaConf.load(config_path)
    model = load_siamese_checkpoint(checkpoint_path, device, cfg)

    print(f"\n[Data] Building {split} dataset...")
    dataset = SiameseXViewDataset(
        root_dir=cfg.data.root_dir,
        cfg=cfg,
        mode=split,
        transform=build_siamese_val_aug(cfg),   # no-op aug, keeps API parity
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.training.get("eval_batch_size", cfg.training.batch_size),
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )

    results = evaluate_siamese(model, loader, device, output_dir)

    print("\n✓ Evaluation complete!")
    print(f"  Results saved to: {output_dir}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the Siamese Attention U-Net on xView2"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/siamese_unet/UNet.pth",
    )
    parser.add_argument(
        "--config", type=str,
        default="configs/siamese_unet_config.yaml",
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="evaluation_results/siamese_unet",
    )

    args = parser.parse_args()
    main(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        split=args.split,
        output_dir=args.output_dir,
    )
