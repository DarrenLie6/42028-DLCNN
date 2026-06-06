# Semantic Segmentation for Building Damage Detection on xView2

## Overview

This module implements **semantic segmentation** for per-pixel damage classification in disaster imagery.

- **Single Input**: Post-disaster RGB images (3 channels)
- **Outputs**: Per-pixel damage classification maps
- **Model**: DeepLabV3 with ResNet-50 backbone
- **xView2 Optimized**: Converts polygon annotations to semantic masks

## Architecture

### DeepLabV3 with ResNet-50

```
Post-disaster Image (512×512 or larger)
    ↓
ResNet-50 Backbone (ImageNet pretrained)
    ↓
ASPP (Atrous Spatial Pyramid Pooling)
    ├─→ Multi-scale atrous convolutions (rates: 6, 12, 18)
    ├─→ Image-level pooling + conv
    ├─→ Concatenate all branches
    ↓
Decoder
    ├─→ Upsample 4× to 1/4 resolution
    ├─→ Combine with low-level features from ResNet
    ├─→ Upsample 4× to original resolution
    ↓
Output: (B, num_classes, H, W) logits
  - Class 0: Background
  - Class 1: Intact buildings
  - Class 2: Damaged buildings
  - Class 3: Destroyed buildings
```

## Per-Pixel Semantic Labels

Each pixel is assigned one of 4 damage classes:

```
0: Background     - Non-building areas
1: Intact         - Buildings with no visible damage
2: Damaged        - Buildings with structural damage
3: Destroyed      - Completely destroyed buildings
-100: Ignore      - Pixels to ignore in loss computation
```

### Label Generation from Polygons

xView2 provides polygon annotations. The dataset:
1. Reads JSON polygon coordinates and damage labels
2. Rasterizes polygons to per-pixel masks
3. Assigns damage class to each pixel inside polygon
4. Handles overlapping polygons (last wins)
5. Resizes to standard tile size (512×512)

## Training

### Quick Start

```bash
python -m src.models.deeplabv3.train_deeplabv3 \
    --config configs/deeplabv3_config.yaml
```

### Configuration

Key parameters in `configs/deeplabv3_config.yaml`:

```yaml
data:
  root_dir:       data/xView2/geotiffs
  tile_size:      512
  num_classes:    4

training:
  batch_size:     4
  learning_rate:  1.0e-4
  epochs:         100
  patience:       15
  checkpoint_dir: checkpoints/semantic_seg
```

### Python API

```python
from src.models.deeplabv3 import build_semantic_model, SemanticSegmentationXViewDataset
from src.models.deeplabv3.deeplabv3_trainer import SemanticSegmentationTrainer
from torch.utils.data import DataLoader

# Build model
model = build_semantic_model(num_classes=4, pretrained=True, device="cuda")

# Build dataset
train_dataset = SemanticSegmentationXViewDataset(
    root_dir="data/xView2/geotiffs",
    cfg=cfg,
    mode="train",
    transform=train_aug,
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Build trainer
trainer = SemanticSegmentationTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device="cuda",
    num_epochs=100,
)

# Train
history = trainer.fit()
```

## Evaluation

### Metrics Computed

- **IoU (Intersection over Union)**: Per-class and mean
- **F1 Score**: Per-class and mean
- **Accuracy**: Per-class and mean
- **Precision & Recall**: Per-class
- **Confusion Matrix**: All class pairs

### Running Evaluation

```bash
python -m src.models.deeplabv3.evaluate_semantic_seg \
    --checkpoint checkpoints/semantic_seg/best.pth \
    --config configs/deeplabv3_config.yaml \
    --split val \
    --output-dir evaluation_results
```

### Output Files

The evaluation script generates:

1. **metrics.json**: Numeric metrics
2. **confusion_matrix.png**: Confusion matrix heatmap
3. **per_class_metrics.png**: Bar charts of IoU, F1, Accuracy
4. **predictions_summary.txt**: Per-sample IoU analysis

Example output:
```
[Results] Per-Class Metrics:
------------------------------------------------------------
Class               IoU        F1      Acc
------------------------------------------------------------
Background      0.8234    0.8901    0.9123
Intact          0.6543    0.7234    0.7856
Damaged         0.4321    0.5234    0.6234
Destroyed       0.3456    0.4123    0.5123
------------------------------------------------------------
Mean            0.5639    0.6373    0.7084
------------------------------------------------------------
```

## File Structure

```
src/models/deeplabv3/
├── deeplabv3.py                    # DeepLabV3 model
├── deeplabv3_dataset.py            # xView2 dataset adapter
├── deeplabv3_trainer.py            # Training loop
├── train_deeplabv3.py              # Training script
├── evaluate_semantic_seg.py         # Evaluation script
├── __init__.py                     # Module exports
├── DEEPLABV3_GUIDE.md              # This file
└── QUICK_START.md                  # Quick start guide

configs/
└── deeplabv3_config.yaml           # Configuration
```

## Performance Tips

1. **GPU Memory**: Use batch_size=4 or smaller if OOM
2. **Training Time**: ~2-3 hours per 100 epochs on V100
3. **Pretrained Weights**: Always use (COCO pretrained ResNet-50)
4. **Augmentation**: Enabled by default (flips, rotations, color jitter)
5. **Learning Rate**: Default 1e-4 works well; try 5e-5 if unstable

## Troubleshooting

### Out of Memory

```python
# Reduce batch size in config
training:
  batch_size: 2  # Instead of 4
```

### Poor Performance / Low IoU

- Train for longer (100+ epochs minimum)
- Check data augmentation is applied
- Verify class weights: class 3 (destroyed) is rare
- Try lower learning rate: 5e-5

### Training is Slow

- Increase num_workers in DataLoader
- Use mixed precision (FP16) - already enabled
- Consider DataParallel for multi-GPU

## Related Modules

- `src/data/xview2_dataset.py` - Base xView2 loading
- `src/training/losses.py` - Custom loss functions (CE, Dice, Focal)
- `src/training/metrics.py` - Evaluation metrics
- `src/data/augmentation_utils.py` - Albumentations pipeline

## References

- **DeepLabV3**: Chen, L. C., et al. (2017)
- **xView2 Dataset**: Gupta, R., et al. (2019)
- **PyTorch Segmentation**: https://github.com/pytorch/vision/tree/main/torchvision/models/segmentation
