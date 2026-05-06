import argparse
import os
import sys
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch
import yaml
import warnings

from src.data.dataloader import get_dataloaders
from src.models.simple_unet import UNet
from src.training.trainer import Trainer

sys.path.insert(0, str(Path(__file__).resolve().parent))
# warnings.filterwarnings("ignore", category=UserWarning)
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["GDAL_PAM_ENABLED"] = "NO"
os.environ["CPL_LOG"] = "nul" if sys.platform == "win32" else "/dev/null"

"""
Launch script for BRIGHT Siamese UNet training.

    - python train.py
    - python train.py --config configs/train_config.yaml
    - python train.py --resume checkpoints/best_model.pth
"""

# reproducability
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# config loader
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    def to_ns(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: to_ns(v) for k, v in d.items()})
        return d

    return to_ns(raw)
    
# device check
def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f" GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(" No GPU found — running on CPU")
    return device

# model
# def build_model(cfg: SimpleNamespace) -> torch.nn.Module:
#     model = SiameseUNet(
#         num_classes = cfg.data.num_classes,
#         # pretrained  = cfg.model.pretrained,
#         # backbone= cfg.model.backbone
#         dropout_p= cfg.model.dropout_p
#     )
#     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f" Model params: {n_params / 1e6:.2f}M")
#     return model
def build_model(cfg) -> torch.nn.Module:
    model = UNet(
        num_classes = cfg.data.num_classes,
        dropout_p   = cfg.model.dropout_p,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params / 1e6:.2f}M")
    return model

# mian

def main():
    parser = argparse.ArgumentParser(description="Train BRIGHT Siamese UNet")
    parser.add_argument(
        "--config",
        type    = str,
        default = "configs/train_config.yaml",
        help    = "Path to YAML config file",
    )
    parser.add_argument(
        "--resume",
        type    = str,
        default = None,
        help    = "Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--seed",
        type    = int,
        default = 42,
        help    = "Random seed for reproducibility",
    )
    args = parser.parse_args()

    # setup
    set_seed(args.seed)
    cfg    = load_config(args.config)
    device = get_device()
    t      = cfg.training

    print(f"\n{'='*60}")
    print(f"  Disaster Assessment — SimpleUNet (Post-Disaster Only)")
    print(f"  Config  : {args.config}")
    print(f"  Device  : {device}")
    print(f"  Epochs  : {t.epochs}")
    print(f"  Batch   : {t.batch_size}")
    print(f"  LR      : {t.learning_rate}")
    print(f"{'='*60}\n")

    # dataloaders
    # get_dataloaders returns (train_loader, val_loader, test_loader)
    train_loader, val_loader, _ = get_dataloaders(cfg)

    # model 
    model = build_model(cfg)
    
    # model = model.to(device)
    
    # dummy_opt = torch.zeros(1, 3, 512, 512).to(device)
    # dummy_sar = torch.zeros(1, 1, 512, 512).to(device)
    # dummy_valid = torch.ones(1, dtype=torch.bool).to(device)
    # with torch.no_grad():
    #     out = model(dummy_opt, dummy_sar, dummy_valid)
    # print(f" Smoke test passed — output shape: {out.shape}")
    
    # print(f"[DEBUG] tile_size from cfg: {cfg.data.tile_size}")
    # print(f"[DEBUG] dummy_opt shape: {dummy_opt.shape}")

    # with torch.no_grad():
    #     out = model(dummy_opt, dummy_sar, dummy_valid)

    # print(f"[DEBUG] output shape: {out.shape}")

    # trainer 
    trainer = Trainer(
        model          = model,
        train_loader   = train_loader,
        val_loader     = val_loader,
        device         = device,
        lr             = t.learning_rate,
        weight_decay   = t.weight_decay,
        num_epochs     = t.epochs,
        patience       = t.patience,
        checkpoint_dir = t.checkpoint_dir,
        t_max          = t.epochs,
        eta_min        = t.min_lr,
    )

    # resume 
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f" Resuming from epoch {start_epoch + 1}")
    else:
        # ── Auto-detect checkpoint if no --resume flag given ──────────────
        auto_ckpt = Path(t.checkpoint_dir) / "UNet.pth"
        if auto_ckpt.exists():
            start_epoch = trainer.load_checkpoint(str(auto_ckpt))
            print(f"Auto-resumed from epoch {start_epoch + 1}")

    # train
    history = trainer.fit()
    
    
    # plot curves and cm
    trainer.plot_history(save_dir=cfg.training.checkpoint_dir)

    # final summary
    best = max(history, key=lambda x: x["mean_iou"])
    print(f"\n{'='*60}")
    print(f"  Best epoch : {best['epoch']}")
    print(f"  Val mIoU   : {best['mean_iou']:.4f}")
    print(f"  Val mF1    : {best.get('val/mean_f1', 0.0):.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

