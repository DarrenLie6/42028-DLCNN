"""
Evaluation script for the transformer-based DeepLabV3+ (MiT/PVTv2 encoder).

This is a *separate* evaluator from `evaluate_semantic_seg.py` (which targets the
ResNet-50 DeepLabV3 baseline) so the two model families never get confused. It
reuses the baseline's model-agnostic reporting helpers — they only consume
`outputs["out"]`, which both models produce — and swaps in a transformer-aware
checkpoint loader.

Outputs (saved to --output-dir):
  - metrics.json              : per-class + mean IoU / F1 / Accuracy
  - confusion_matrix.png      : 4x4 GT-vs-prediction heatmap
  - per_class_metrics.png     : IoU / F1 / Accuracy bar charts
  - sample_comparisons/       : image | ground-truth overlay | prediction overlay
  - predictions_summary.txt   : per-sample pixel-agreement table

Usage:
    python -m src.models.deeplabv3.evaluate_deeplabv3plus_transformer 
        --checkpoint checkpoints/semantic_seg_transformer/<best>.pth 
        --config configs/deeplabv3plus_transformer_config.yaml 
        --split test 
        --output-dir evaluation_results/deeplabv3plus_transformer
"""

from __future__ import annotations
import argparse
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.deeplabv3 import SemanticSegmentationXViewDataset
from src.models.deeplabv3.deeplabv3plus_transformer import build_transformer_semantic_model
from src.data.siamese_xview2_dataset import SiameseXViewDataset
from src.training.metrics import SegmentationMetrics

# Reuse the baseline's model-agnostic evaluation + plotting pipeline.
from src.models.deeplabv3.evaluate_semantic_seg import (
    evaluate, save_results, NUM_CLASSES, IGNORE_INDEX, LABEL_NAMES,
)


def evaluate_bitemporal(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    save_dir: str,
) -> dict:
    """
    Bi-temporal variant of the shared `evaluate()`.

    Identical metrics/reporting, but the input is a 6-channel [pre|post] tensor,
    so for visualisation we keep only the POST-disaster channels (3:6) — the
    shared evaluator's 3-channel imshow path cannot render 6 channels.
    """
    model.eval()
    metrics = SegmentationMetrics(
        num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX, device=device,
    )
    all_predictions = []

    print("\n[Evaluation] Running inference (bi-temporal)...")
    with torch.no_grad():
        for batch in tqdm(data_loader):
            images = batch["image"].to(device)   # (B, 6, H, W) = [pre | post]
            labels = batch["label"].to(device)
            stems  = batch["stem"]

            logits = model(images)["out"]
            metrics.update(logits, labels)
            preds = logits.argmax(dim=1)

            for i, stem in enumerate(stems):
                # Visualise the POST image only (channels 3:6), already in [0,1].
                post_np = images[i, 3:].cpu().numpy().transpose(1, 2, 0)
                post_np = np.clip(post_np, 0.0, 1.0)
                all_predictions.append({
                    "stem":   stem,
                    "image":  post_np,
                    "pred":   preds[i].cpu().numpy(),
                    "gt":     labels[i].cpu().numpy(),
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


def load_transformer_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    cfg,
) -> torch.nn.Module:
    """
    Rebuild the transformer DeepLabV3+ exactly as it was trained (same encoder
    structure) and load the trained weights.

    The encoder structure MUST match the checkpoint: a model trained with the
    timm PVTv2 backbone has `encoder.backbone.*` keys, while the from-scratch
    MiT encoder has `encoder.patch_embeds.*` keys. We therefore rebuild using
    the same `variant` / `pretrained` / `timm_backbone` settings as in the
    training config.
    """
    print(f"[Loading] {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    variant       = cfg.model.get("variant", "b1")
    aux_loss      = cfg.model.get("aux_loss", True)
    pretrained    = cfg.model.get("pretrained", True)
    timm_backbone = cfg.model.get("timm_backbone", None)
    bitemporal    = bool(cfg.model.get("bitemporal", False))

    model = build_transformer_semantic_model(
        num_classes=NUM_CLASSES,
        variant=variant,
        aux_loss=aux_loss,
        pretrained=pretrained,        # rebuilds the SAME encoder structure
        timm_backbone=timm_backbone,
        bitemporal=bitemporal,        # MUST match training (adds fusion blocks)
        device=device,
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # A large number of missing keys means the rebuilt structure does not match
    # the checkpoint (e.g. trained with timm but timm is now unavailable, so the
    # builder fell back to the from-scratch encoder). Fail loudly rather than
    # silently evaluating partially-random weights.
    n_model = sum(1 for _ in model.state_dict())
    if len(missing) > 0.2 * n_model:
        raise RuntimeError(
            f"Checkpoint does not match the rebuilt model: {len(missing)}/{n_model} "
            f"parameters were not loaded.\n"
            f"This usually means the encoder structure differs from training "
            f"(e.g. the checkpoint used the timm PVTv2 backbone but `timm` is not "
            f"available now, so a from-scratch MiT encoder was built instead).\n"
            f"Fix: ensure `timm` is installed and `model.pretrained`/`model.variant`/"
            f"`model.timm_backbone` in the config match how the checkpoint was trained.\n"
            f"  Missing (sample):    {missing[:4]}\n"
            f"  Unexpected (sample): {unexpected[:4]}"
        )

    if missing:
        print(f"  [WARN] Missing keys ({len(missing)}):    {missing[:6]}{' ...' if len(missing) > 6 else ''}")
    if unexpected:
        print(f"  [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:6]}{' ...' if len(unexpected) > 6 else ''}")

    model.eval()

    epoch    = checkpoint.get("epoch", "?")
    mean_iou = checkpoint.get("mean_iou", "?")
    kind     = getattr(model, "encoder_kind", "?")
    print(f"  Encoder: {kind} ({cfg.model.get('variant', 'b1')}) | "
          f"Epoch: {epoch} | Train-time best mIoU: {mean_iou}")

    return model


def main(
    checkpoint_path: str,
    config_path: str,
    split: str = "test",
    output_dir: str = "evaluation_results/deeplabv3plus_transformer",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")

    cfg   = OmegaConf.load(config_path)
    model = load_transformer_checkpoint(checkpoint_path, device, cfg)

    bitemporal = bool(cfg.model.get("bitemporal", False))
    print(f"\n[Data] Building {split} dataset (bitemporal={bitemporal})...")
    # Bi-temporal models need the 6-channel [pre|post] loader; single-input
    # models use the post-only loader. No augmentation at eval either way.
    dataset_cls = SiameseXViewDataset if bitemporal else SemanticSegmentationXViewDataset
    dataset = dataset_cls(
        root_dir=cfg.data.root_dir,
        cfg=cfg,
        mode=split,
        transform=None,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.training.get("eval_batch_size", 4),
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )

    # Bi-temporal input (6ch) needs the post-only-visualisation path; the shared
    # 3-channel evaluate() handles the single-input case.
    if bitemporal:
        results = evaluate_bitemporal(model, loader, device, output_dir)
    else:
        results = evaluate(model, loader, device, output_dir)

    print("\n✓ Evaluation complete!")
    print(f"  Results saved to: {output_dir}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the transformer DeepLabV3+ on xView2"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--config", type=str,
        default="configs/deeplabv3plus_transformer_config.yaml",
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="evaluation_results/deeplabv3plus_transformer",
    )

    args = parser.parse_args()
    main(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        split=args.split,
        output_dir=args.output_dir,
    )
