"""Configuration for YOLO11 xBD segmentation model."""

import os

# ─────────────────────────────────────────────────────────────────────────────
# Class Configuration
# ─────────────────────────────────────────────────────────────────────────────

SEG_CLASSES  = ['background', 'intact', 'damaged', 'destroyed']
NUM_CLASSES  = 4

# YOLO seg uses 0-indexed, background is NOT a YOLO class (YOLO ignores BG)
# We only annotate the 3 building classes for YOLO polygon seg format
YOLO_CLASSES = ['intact', 'damaged', 'destroyed']   # class 0, 1, 2

XBD_TO_YOLO = {
    'no-damage':    0,   # intact
    'minor-damage': 1,   # damaged (merged)
    'major-damage': 1,   # damaged (merged)
    'destroyed':    2,   # destroyed
}

CLASS_COLORS = {
    0: (0,   0,   0),    # background — black
    1: (0,   200, 0),    # intact     — green
    2: (255, 165, 0),    # damaged    — orange
    3: (220, 0,   0),    # destroyed  — red
}

# ─────────────────────────────────────────────────────────────────────────────
# Dataset & Paths
# ─────────────────────────────────────────────────────────────────────────────

BASE_DATA_DIR = r"E:\UTS\CNN and Deep Learning\Assignment 3\42028-DLCNN\data\xView2\geotiffs"
YOLO_DS_DIR   = r"E:\UTS\CNN and Deep Learning\Assignment 3\42028-DLCNN\xbd_yolo"
IMG_SIZE      = 640   # YOLO standard

TRAIN_DIRS = [
    os.path.join(BASE_DATA_DIR, "tier1"),
    os.path.join(BASE_DATA_DIR, "tier3"),
]
VAL_DIR    = os.path.join(BASE_DATA_DIR, "hold")
TEST_DIR   = os.path.join(BASE_DATA_DIR, "test")

# ─────────────────────────────────────────────────────────────────────────────
# Training Configuration
# ─────────────────────────────────────────────────────────────────────────────

TRAINING_CONFIG = {
    'epochs':        150,
    'imgsz':         IMG_SIZE,
    'batch':         16,
    'device':        0,                    # GPU 0
    'optimizer':     "AdamW",
    'lr0':           1e-4,
    'lrf':           0.01,
    'momentum':      0.937,
    'weight_decay':  1e-4,
    'warmup_epochs': 3,
    'cos_lr':        True,                 # cosine LR schedule
    'hsv_h':         0.015,                # colour augmentation
    'hsv_s':         0.7,
    'hsv_v':         0.4,
    'flipud':        0.5,
    'fliplr':        0.5,
    'degrees':       15.0,                 # rotation aug (satellite imagery)
    'translate':     0.1,
    'scale':         0.5,
    'val':           True,
    'save':          True,                 # Save weights
    'save_period':   1,                    # Save checkpoint every 1 epoch
    'plots':         True,                 # Plot results
    'verbose':       True,
    'patience':      50,                   # Early stopping patience
}

# ─────────────────────────────────────────────────────────────────────────────
# Model Configuration
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "yolo11l-seg.pt"  # l=large; use yolo11x-seg.pt for max accuracy
RUN_NAME = "xbd_seg_v1"

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Configuration
# ─────────────────────────────────────────────────────────────────────────────

EVAL_CONFIG = {
    'imgsz':   IMG_SIZE,
    'batch':   16,
    'device':  0,
    'verbose': True,
    'plots':   True,
    'save_json': True,
}

# Inference configuration
INFERENCE_CONFIG = {
    'imgsz':  IMG_SIZE,
    'conf':   0.25,
    'iou':    0.45,
    'device': 0,
    'save':   False,
    'verbose': False,
}
