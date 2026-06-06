# Quick Start Guide: Semantic Segmentation on xView2

## 5-Minute Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- torch, torchvision 2.2.0+
- rasterio (GeoTIFF support)
- opencv-python
- albumentations (augmentation)
- omegaconf (config management)
- matplotlib, seaborn (visualization)

### 2. Prepare Data

Ensure xView2 dataset structure:

```
data/xView2/geotiffs/
├── tier1/          # Training images
│   ├── guatemala-volcano_00000000_post_disaster.tif
│   ├── guatemala-volcano_00000000_post_disaster.json
│   └── ...
├── tier3/          # Additional training images
│   └── ...
├── hold/           # Validation images
│   └── ...
└── test/           # Test images
    └── ...
```

The dataset loader automatically finds paired `.tif` and `.json` files.

### 3. Train Model

```bash
# Using default config
python -m src.models.deeplabv3.train_deeplabv3

# Or specify custom config
python -m src.models.deeplabv3.train_deeplabv3 \
    --config configs/deeplabv3_config.yaml
```

**Expected output:**
```
[Device] Using cuda
[Data] Building datasets...
  Train: 521 samples
  Val:   88 samples
[Model] Building DeepLabV3...
  Trainable params: 39,124,304
[Trainer] Initializing...
[Training] Starting...
Train 1: 100%|████████| 65/65 [02:34<00:00,  2.38s/it]
Val 1: 100%|████████| 22/22 [00:18<00:00,  1.19s/it]
[Epoch 1] train_loss=0.3854 | val_loss=0.2943 | val_mIoU=0.5432 | lr=1.00e-04 | time=173.2s
```

**Training time:** ~2-3 hours for 100 epochs on V100 GPU

### 4. Evaluate Model

```bash
# Find best checkpoint
ls checkpoints/semantic_seg/

# Run evaluation on validation set
python -m src.models.deeplabv3.evaluate_semantic_seg \
    --checkpoint checkpoints/semantic_seg/semantic_seg_best_mIoU_0.7234_epoch_45.pth \
    --config configs/deeplabv3_config.yaml \
    --split val \
    --output-dir evaluation_results
```

**Output files:**
- `evaluation_results/metrics.json` - Numeric metrics
- `evaluation_results/confusion_matrix.png` - Class confusion matrix
- `evaluation_results/per_class_metrics.png` - IoU/F1/Accuracy charts
- `evaluation_results/predictions_summary.txt` - Per-sample analysis

### 5. View Results

Example metrics.json:
```json
{
  "mean_iou": 0.6543,
  "mean_f1": 0.7234,
  "mean_acc": 0.8123,
  "iou/Background": 0.8901,
  "iou/Intact": 0.6543,
  "iou/Damaged": 0.4321,
  "iou/Destroyed": 0.3456,
  ...
}
```

## Common Use Cases

### Use Case 1: Load Trained Model

```python
import torch
from src.models.deeplabv3 import build_semantic_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model
model = build_semantic_model(num_classes=4, pretrained=False, device=device)

# Load checkpoint
checkpoint = torch.load("checkpoints/semantic_seg/best.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("✓ Model loaded successfully!")
```

### Use Case 2: Predict on Single Image

```python
import cv2
import rasterio
import numpy as np
from torchvision import transforms
import torch

# Load GeoTIFF
with rasterio.open("image.tif") as src:
    image = src.read([1, 2, 3]).transpose(1, 2, 0)  # (H, W, 3)

# Normalize to [0, 1]
image = image.astype(np.float32) / 255.0

# Convert to tensor
image_tensor = torch.from_numpy(image).permute(2, 0, 1)  # (3, H, W)
image_tensor = image_tensor.unsqueeze(0).to(device)  # (1, 3, H, W)

# Predict
with torch.no_grad():
    outputs = model(image_tensor)
    logits = outputs['out']
    pred = logits.argmax(dim=1)  # (1, H, W)

prediction = pred[0].cpu().numpy()  # (H, W)

# Class labels
class_names = {0: "Background", 1: "Intact", 2: "Damaged", 3: "Destroyed"}
print(f"Unique classes: {[class_names[c] for c in np.unique(prediction)]}")
```

### Use Case 3: Batch Evaluation on Test Set

```python
from torch.utils.data import DataLoader
from src.models.deeplabv3 import SemanticSegmentationXViewDataset
from src.training.metrics import SegmentationMetrics

# Create test dataset
test_dataset = SemanticSegmentationXViewDataset(
    root_dir="data/xView2/geotiffs",
    cfg=cfg,
    mode="test",
    transform=None,
)

test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Compute metrics
metrics = SegmentationMetrics(num_classes=4, device=device)
model.eval()

with torch.no_grad():
    for batch in test_loader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(images)
        logits = outputs['out']
        
        metrics.update(logits, labels)

results = metrics.compute()
print(f"Test Mean IoU: {results['mean_iou']:.4f}")
print(f"Test Mean F1:  {results['mean_f1']:.4f}")
```

### Use Case 4: Fine-tune on Custom Data

```python
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Load pretrained model
model = build_semantic_model(num_classes=4, pretrained=True, device=device)

# Freeze backbone, only train head
for name, param in model.backbone.named_parameters():
    param.requires_grad = False

# Setup training
optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# Train on custom data...
```

## Configuration Reference

### Default Config Location

`configs/deeplabv3_config.yaml`:

```yaml
data:
  root_dir: data/xView2/geotiffs
  tile_size: 512
  num_classes: 4

training:
  batch_size: 4
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4
  epochs: 100
  patience: 15
  checkpoint_dir: checkpoints/semantic_seg
  min_lr: 1.0e-6
  num_workers: 4

augmentation:
  horizontal_flip_p: 0.5
  vertical_flip_p: 0.5
  rotate_90_p: 0.25
  brightness_contrast:
    brightness_limit: 0.2
    contrast_limit: 0.2
  gaussian_noise_p: 0.1
  elastic_transform_p: 0.1
  coarse_dropout_p: 0.1
```

### Modify Configuration

Create `configs/custom_config.yaml`:

```yaml
data:
  root_dir: /custom/path/to/data
  tile_size: 512
  num_classes: 4

training:
  batch_size: 2  # Reduced for smaller GPU
  learning_rate: 5.0e-5  # Lower LR
  epochs: 200  # Train longer
  patience: 20
  checkpoint_dir: checkpoints/custom_run
```

Train with custom config:

```bash
python -m src.models.deeplabv3.train_deeplabv3 --config configs/custom_config.yaml
```

## Troubleshooting

### Error: "CUDA out of memory"

**Solution**: Reduce batch size

```yaml
training:
  batch_size: 2  # or 1 for very limited GPU
```

### Error: "FileNotFoundError: data/xView2/geotiffs"

**Solution**: Update `data.root_dir` in config to your xView2 path

### Training loss not decreasing

**Possible causes**:
1. Learning rate too high → Use `5.0e-5` instead of `1.0e-4`
2. Data not loading correctly → Check console output for warnings
3. Model not initialized properly → Ensure `pretrained=True` in config

### Model predicting mostly one class

**Likely issue**: Class imbalance. The class weights in `src/models/deeplabv3/deeplabv3_trainer.py` may need adjustment:

```python
CLASS_WEIGHTS = [0.5, 5.0, 7.0, 10.0]  # Background, Intact, Damaged, Destroyed
```

Increase weights for underrepresented classes.

## Next Steps

- 📖 Read [DEEPLABV3_GUIDE.md](./DEEPLABV3_GUIDE.md) for detailed architecture explanation
- 🎯 Try [Use Case 3](#use-case-3-batch-evaluation-on-test-set) to evaluate on your dataset
- 🔧 Experiment with different configs in `configs/`
- 📊 Compare checkpoints: `checkpoints/semantic_seg/`

## References

- Dataset: [xView2 Challenge](https://www.drivendata.org/competitions/60/nasa-tropical/)
- Model: [DeepLabV3 Paper](https://arxiv.org/abs/1706.05587)
- Implementation: [PyTorch Segmentation Models](https://github.com/pytorch/vision/tree/main/torchvision/models/segmentation)
