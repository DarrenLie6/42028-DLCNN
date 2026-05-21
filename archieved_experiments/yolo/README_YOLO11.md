# YOLO11 xBD Segmentation - Training & Evaluation

This directory contains standalone Python scripts for training and evaluating a YOLO11 segmentation model on the xBD (xView Building Damage) dataset.

## Overview

The scripts extracted from the `yolo11.ipynb` notebook provide:

1. **config.py** - Configuration and constants for the YOLO11 model
2. **train.py** - Training script for the YOLO11 segmentation model
3. **evaluate.py** - Evaluation and visualization script

## Project Structure

```
yolo/
├── yolo11.ipynb          # Original Jupyter notebook
├── config.py             # Configuration & constants
├── train.py              # Training script
├── evaluate.py           # Evaluation & visualization script
└── README.md             # This file
```

## Dataset Structure

The scripts expect the xBD dataset to be organized as follows:

```
xbd_yolo/
├── images/
│   ├── train/  # Training images
│   ├── val/    # Validation images
│   └── test/   # Test images
├── labels/
│   ├── train/  # Training labels (YOLO polygon format)
│   ├── val/    # Validation labels
│   └── test/   # Test labels
└── xbd.yaml    # Dataset YAML (auto-generated)
```

## Configuration

Edit `config.py` to customize:

- **Paths**: `BASE_DATA_DIR`, `YOLO_DS_DIR`
- **Training parameters**: `TRAINING_CONFIG` (epochs, batch size, learning rate, etc.)
- **Model**: `MODEL_NAME` (yolo11l-seg.pt, yolo11x-seg.pt, etc.)
- **Classes**: `YOLO_CLASSES` (intact, damaged, destroyed)

## Training

### Quick Start

```bash
python train.py
```

This will:
1. Generate the dataset YAML configuration
2. Load the YOLO11-Large segmentation model
3. Train for 150 epochs with the configured parameters
4. Save checkpoints and results to `xbd_yolo/runs/xbd_seg_v1/`

### Training Configuration

Key parameters in `config.py`:

```python
'epochs':        150,        # Number of epochs
'batch':         4,          # Batch size
'imgsz':         640,        # Image size
'device':        0,          # GPU device index
'optimizer':     "AdamW",    # Optimizer
'lr0':           1e-4,       # Initial learning rate
'weight_decay':  1e-4,       # L2 regularization
'cos_lr':        True,       # Cosine annealing schedule
```

### Outputs

Training results are saved to: `xbd_yolo/runs/xbd_seg_v1/`

```
xbd_seg_v1/
├── weights/
│   ├── best.pt     # Best model weights
│   └── last.pt     # Last epoch weights
├── results.csv     # Training metrics per epoch
├── confusion_matrix.png
├── labels.jpg
└── ... (other YOLO outputs)
```

## Evaluation

### Quick Start

```bash
python evaluate.py
```

This will:
1. Load the best trained model weights
2. Run inference on validation images
3. Generate damage heatmaps (individual & aggregate)
4. Calculate mAP50, mAP50-95, and per-class metrics

### Evaluation Features

1. **Inference**: Runs prediction on 50 validation images
2. **Heatmaps**: Visualizes damage predictions as confidence heatmaps
3. **Aggregation**: Creates an aggregate heatmap showing spatial damage distribution
4. **Metrics**: Computes official YOLO validation metrics (mAP, per-class AP)

### Outputs

Evaluation results are saved to:

- **Heatmaps**: `xbd_yolo/heatmaps/`
  - Individual heatmaps for sample scenes
  - Aggregate heatmap across all scenes
- **Metrics**: Printed to console and saved by YOLO

## Classes

The model predicts 3 building damage classes:

| Class | Label | Color |
|-------|-------|-------|
| 0 | Intact | Green |
| 1 | Damaged | Orange |
| 2 | Destroyed | Red |

## Model Architectures

Available YOLO11 segmentation models:

- `yolo11n-seg.pt` - Nano (fastest, least accurate)
- `yolo11s-seg.pt` - Small
- `yolo11m-seg.pt` - Medium
- `yolo11l-seg.pt` - Large (default)
- `yolo11x-seg.pt` - Extra Large (slowest, most accurate)

Change in `config.py`:
```python
MODEL_NAME = "yolo11x-seg.pt"  # For maximum accuracy
```

## Advanced Usage

### Custom Training Parameters

Modify `TRAINING_CONFIG` in `config.py`:

```python
TRAINING_CONFIG = {
    'epochs':        200,      # More epochs
    'batch':         8,        # Larger batch
    'lr0':           5e-5,     # Different learning rate
    'patience':      50,       # Early stopping patience
    # ... other parameters
}
```

### Custom Inference Parameters

Modify `INFERENCE_CONFIG` in `config.py`:

```python
INFERENCE_CONFIG = {
    'conf':   0.50,    # Higher confidence threshold
    'iou':    0.50,    # Different NMS IoU threshold
    # ... other parameters
}
```

### Running Inference on Test Set

In `evaluate.py`:

```python
pred_results = run_inference(model, split="test", num_samples=100)
```

## Dependencies

Required packages (from `requirements.txt`):

```
ultralytics
supervision
matplotlib
seaborn
tifffile
shapely
opencv-python
torch
torchvision
```

Install with:
```bash
pip install -r ../requirements.txt
```

## Output Descriptions

### Heatmaps

Each heatmap figure contains 5 panels:

1. **Original Image**: Post-disaster RGB image
2. **Intact Heatmap**: Confidence scores for undamaged buildings
3. **Damaged Heatmap**: Confidence scores for partially damaged buildings
4. **Destroyed Heatmap**: Confidence scores for destroyed buildings
5. **Composite Overlay**: All classes blended onto original image

Colors indicate confidence strength (lighter = higher confidence).

### Aggregate Heatmap

Shows:
- **Panel 1**: Spatial distribution of intact buildings
- **Panel 2**: Spatial distribution of damaged buildings
- **Panel 3**: Spatial distribution of destroyed buildings
- **Panel 4**: Severity score (destroyed − intact) for disaster impact assessment

### Metrics

```
mAP@50      - Mean Average Precision at IoU=0.50
mAP@50-95   - Mean Average Precision at IoU=0.50:0.95
Per-class AP - Average Precision for each damage class
```

## Troubleshooting

### CUDA/GPU Issues

Ensure GPU is available:
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.current_device())
```

Change device in `config.py`:
```python
'device': 'cpu',  # Use CPU if GPU unavailable
```

### Out of Memory (OOM)

Reduce batch size in `config.py`:
```python
'batch': 2,  # Smaller batch size
```

Or use a smaller model:
```python
MODEL_NAME = "yolo11m-seg.pt"  # Medium instead of Large
```

### Missing Dataset

Ensure xBD dataset is properly converted to YOLO format. The conversion notebook step should create:
- `xbd_yolo/images/{train,val,test}/`
- `xbd_yolo/labels/{train,val,test}/`

## References

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [xView Building Damage Dataset](https://xviewdataset.org/)

## Author Notes

These scripts were extracted from `yolo11.ipynb` to provide:
- Modular, reusable code
- Easier debugging and production deployment
- Clear separation of concerns (config → train → evaluate)
- Extensibility for custom modifications

Both training and evaluation can be run independently after model files are available.
