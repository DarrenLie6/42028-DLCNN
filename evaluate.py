import torch
import yaml
from omegaconf import OmegaConf
from torch.amp import autocast
from pathlib import Path
from src.data.dataloader import get_dataloaders
from src.models.siamese_unet import SiameseUNet
from src.training.metrics import SegmentationMetrics
from src.training.losses import CombinedLoss
from src.training.trainer import NUM_CLASSES, LABEL_NAMES, CLASS_WEIGHTS, IGNORE_INDEX
from train import build_model

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import random

SCRIPT_DIR = Path(__file__).resolve().parent

def evaluate(
    cfg_path:  str = str(SCRIPT_DIR / "train_config.yaml"),
    ckpt_path: str = str(SCRIPT_DIR / "checkpoints" / "UNet.pth"),
):

    cfg    = OmegaConf.load(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = get_dataloaders(cfg)

    if test_loader is None:
        print("No test split found.")
        return

    # 
    model = build_model(cfg).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Checkpoint   : {ckpt_path}")
    print(f"Epoch        : {ckpt['epoch']}")
    print(f"Val mIoU     : {ckpt['val_mean_iou']:.4f}\n")

    # ── Reuse your existing loss + metrics ────────────────────────────
    criterion = CombinedLoss(
        class_weights=CLASS_WEIGHTS,
        num_classes=NUM_CLASSES,
        ignore_index=IGNORE_INDEX,
    ).to(device)

    metrics    = SegmentationMetrics(device=device)
    total_loss = 0.0
    n_batches  = 0

    # ── Test loop ─────────────────────────────────────────────────────
    with torch.no_grad():
        for batch in test_loader:
            optical       = batch["optical"].to(device)
            sar           = batch["sar"].to(device)
            targets       = batch["label"].to(device)
            optical_valid = batch["optical_valid"].to(device)

            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits           = model(optical, sar, optical_valid)
                loss, _, _       = criterion(logits, targets)

            total_loss += loss.item()
            n_batches  += 1
            metrics.update(logits, targets)

    # ── Results ───────────────────────────────────────────────────────
    results = metrics.compute()
    
    print("Available metric keys:", list(results.keys()))

    print("─" * 42)
    print(f"  Test Loss     : {total_loss / n_batches:.4f}")
    print(f"  Test mIoU     : {results['mean_iou']:.4f}")
    print(f"  Test mean F1  : {results['mean_f1']:.4f}")
    print(f"  Test mean Acc : {results['mean_acc']:.4f}")
    print("─" * 42)
    
    CLASS_NAMES = ["Background", "Intact", "Damaged", "Destroyed"]

    print("─" * 50)
    print(f"  {'Class':<12} {'IoU':>8}  {'F1':>8}  {'Acc':>8}")
    print("─" * 50)
    for name in CLASS_NAMES[1:]:   # skip Background
        iou = results.get(f"iou/{name}", float("nan"))
        f1  = results.get(f"f1/{name}",  float("nan"))
        acc = results.get(f"acc/{name}", float("nan"))
        print(f"  {name:<12}  {iou:>8.4f}  {f1:>8.4f}  {acc:>8.4f}")
    print("─" * 50)
    print(f"  {'Mean (1-3)':<12}  {results['mean_iou']:>8.4f}  {results['mean_f1']:>8.4f}  {results['mean_acc']:>8.4f}")
    print("─" * 50)
    
        # ── After printing metrics — visualise random test samples ────────
    visualise_samples(
        model       = model,
        test_loader = test_loader,
        device      = device,
        save_dir    = str(SCRIPT_DIR / "checkpoints"),
        n_samples   = 4,     # ← change to see more samples
    )

PALETTE = {
    0: (0.15, 0.15, 0.15),   # Background — dark grey
    1: (0.20, 0.60, 0.20),   # Intact     — green
    2: (1.00, 0.75, 0.00),   # Damaged    — amber
    3: (0.85, 0.15, 0.15),   # Destroyed  — red
}

def labels_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert (H,W) int mask → (H,W,3) RGB using PALETTE."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.float32)
    for cls, colour in PALETTE.items():
        rgb[mask == cls] = colour
    return rgb


def visualise_samples(model, test_loader, device, save_dir: str, n_samples: int = 4):
    """Pick one random batch and save n_samples side-by-side comparisons."""

    model.eval()
    save_path = Path(save_dir) / "test_samples.png"

    # ── Grab one random batch ─────────────────────────────────────────
    batch = random.choice(list(test_loader))

    optical       = batch["optical"].to(device)
    sar           = batch["sar"].to(device)
    optical_valid = batch["optical_valid"].to(device)
    targets       = batch["label"]                     # keep on CPU for plotting

    with torch.no_grad():
        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(optical, sar, optical_valid)
    preds = logits.argmax(dim=1).cpu()                 # (B, H, W)

    # ── Plot ──────────────────────────────────────────────────────────
    n = min(n_samples, optical.size(0))
    fig, axes = plt.subplots(n, 4, figsize=(18, 4 * n))
    fig.suptitle("Test Samples — Optical | SAR | Ground Truth | Prediction", fontsize=13)

    if n == 1:
        axes = axes[np.newaxis, :]   # ensure 2D indexing

    for i in range(n):
        # Optical — denormalise from ImageNet mean/std
        opt_img = optical[i].cpu().permute(1, 2, 0).numpy()
        mean    = np.array([0.485, 0.456, 0.406])
        std     = np.array([0.229, 0.224, 0.225])
        opt_img = (opt_img * std + mean).clip(0, 1)

        # SAR — single channel
        sar_img = sar[i].cpu().squeeze().numpy()
        sar_img = (sar_img - sar_img.min()) / (sar_img.max() - sar_img.min() + 1e-6)

        gt_rgb   = labels_to_rgb(targets[i].numpy())
        pred_rgb = labels_to_rgb(preds[i].numpy())

        axes[i, 0].imshow(opt_img);           axes[i, 0].set_title("Optical")
        axes[i, 1].imshow(sar_img, cmap="gray"); axes[i, 1].set_title("SAR")
        axes[i, 2].imshow(gt_rgb);            axes[i, 2].set_title("Ground Truth")
        axes[i, 3].imshow(pred_rgb);          axes[i, 3].set_title("Prediction")

        for ax in axes[i]:
            ax.axis("off")

    # ── Legend ────────────────────────────────────────────────────────
    legend = [
        mpatches.Patch(color=PALETTE[0], label="Background"),
        mpatches.Patch(color=PALETTE[1], label="Intact"),
        mpatches.Patch(color=PALETTE[2], label="Damaged"),
        mpatches.Patch(color=PALETTE[3], label="Destroyed"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4,
            bbox_to_anchor=(0.5, -0.02), fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Samples saved → {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(SCRIPT_DIR / "configs/train_config.yaml"))
    parser.add_argument("--ckpt",   default=str(SCRIPT_DIR / "checkpoints" / "UNet.pth"))
    args = parser.parse_args()
    evaluate(args.config, args.ckpt)