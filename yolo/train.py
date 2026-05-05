"""
Usage:
    python yolo/train.py

    To resume a stopped run:
        python yolo/train.py --resume
"""

from __future__ import annotations
import argparse
from pathlib import Path
from ultralytics import YOLO


# Config
DATA_YAML   = "bright_yolo/data.yaml"
MODEL       = "yolov8m-seg.pt"   # pretrained COCO checkpoint
PROJECT_DIR = "yolo_bright"
RUN_NAME    = "yolov8m_seg_sar"

def train(resume: bool = False) -> None:
    weights = MODEL
    if resume:
        last = Path(PROJECT_DIR) / RUN_NAME / "weights" / "last.pt"
        if not last.exists():
            raise FileNotFoundError(
                f"No checkpoint found at {last}. "
                "Run without --resume to start fresh."
            )
        weights = str(last)
        print(f"Resuming from {weights}")

    model = YOLO(weights)

    model.train(
        data    = DATA_YAML,
        epochs  = 50,
        imgsz   = 1024,   
        batch   = 4,

        # Augmentation
        fliplr  = 0.5,
        flipud  = 0.5,
        degrees = 90,
        mosaic  = 1.0,
        erasing = 0.3,
        
        cls     = 2.0,

        # Optimiser
        optimizer    = "AdamW",
        lr0          = 1e-3,
        lrf          = 0.01,
        weight_decay = 1e-4,
        warmup_epochs= 3,

        # Logging
        project     = PROJECT_DIR,
        name        = RUN_NAME,
        save_period = 5,
        val         = True,

        device  = 0,
        workers = 4,
        exist_ok= resume,
        verbose = True,
    )

    print(f"\nTraining complete.")
    print(f"Best weights -> {PROJECT_DIR}/{RUN_NAME}/weights/best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from last checkpoint"
    )
    args = parser.parse_args()
    train(resume=args.resume)