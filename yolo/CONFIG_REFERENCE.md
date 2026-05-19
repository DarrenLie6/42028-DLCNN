# Configuration Reference - Checkpoint & Curve Saving

## Updated TRAINING_CONFIG

```python
TRAINING_CONFIG = {
    'epochs':        150,                    # Total training epochs
    'imgsz':         IMG_SIZE,              # Image size (640)
    'batch':         4,                     # Batch size
    'device':        0,                     # GPU device
    'optimizer':     "AdamW",               # Optimizer type
    'lr0':           1e-4,                  # Initial learning rate
    'lrf':           0.01,                  # Final learning rate ratio
    'momentum':      0.937,                 # Momentum
    'weight_decay':  1e-4,                  # L2 regularization
    'warmup_epochs': 3,                     # Warmup epochs
    'cos_lr':        True,                  # Cosine LR annealing
    'hsv_h':         0.015,                 # Hue augmentation
    'hsv_s':         0.7,                   # Saturation augmentation
    'hsv_v':         0.4,                   # Value augmentation
    'flipud':        0.5,                   # Vertical flip probability
    'fliplr':        0.5,                   # Horizontal flip probability
    'degrees':       15.0,                  # Rotation angle
    'translate':     0.1,                   # Translation
    'scale':         0.5,                   # Scale augmentation
    'val':           True,                  # Validate each epoch
    'save':          True,                  # Save weights
    'save_period':   1,                     # ⭐ SAVE CHECKPOINT EVERY 1 EPOCH
    'plots':         True,                  # ⭐ PLOT RESULTS EACH EPOCH
    'verbose':       True,                  # Detailed logging
    'patience':      50,                    # ⭐ EARLY STOPPING PATIENCE
}
```

## Key Parameters Explained

### Checkpoint Saving
- **`save=True`** - Enable weight saving
- **`save_period=1`** - Save after every single epoch
  - Use `save_period=5` to save every 5 epochs
  - Use `save_period=10` to save every 10 epochs

### Training Curves
- **`plots=True`** - Enable plot generation
- Updates `results.png` at end of each epoch
- Saves to training run directory

### Early Stopping
- **`patience=50`** - Stop if validation doesn't improve for 50 epochs
- Saves training time if model plateaus
- Always saves `best.pt` regardless

## Output Files Generated

### During Training (Updated Each Epoch)

| File | Purpose | Updated |
|------|---------|---------|
| `results.csv` | Metrics per epoch | Every epoch |
| `results.png` | 8 training curve plots | Every epoch |
| `epoch{N}.pt` | Checkpoint after epoch N | Every epoch |
| `last.pt` | Latest checkpoint | Every epoch |
| `best.pt` | Best model so far | When validation improves |

### Example Results CSV

```
epoch,train/box_loss,train/seg_loss,train/cls_loss,train/dfl_loss,...
0,2.2264,3.3826,2.0423,1.1854,...
1,2.1472,3.2438,1.6525,1.1582,...
2,2.0890,3.1564,1.5119,1.1230,...
```

### Example Results PNG
Shows 8 subplots:
1. box_loss (training)
2. seg_loss (training)
3. cls_loss (training)
4. dfl_loss (training)
5. Metrics precision
6. Metrics recall
7. box_loss (validation)
8. seg_loss (validation)

## Usage Examples

### Standard Training
```bash
python train.py
```
Saves checkpoints after each epoch.

### Resume from Last Checkpoint
```python
from ultralytics import YOLO
model = YOLO('xbd_yolo/runs/xbd_seg_v1/weights/last.pt')
model.train(data='xbd_yolo/xbd.yaml', epochs=200, resume=True)
```

### Use Specific Epoch Checkpoint
```python
from ultralytics import YOLO
model = YOLO('xbd_yolo/runs/xbd_seg_v1/weights/epoch50.pt')
predictions = model.predict('image.jpg')
```

### Save Every 5 Epochs (Save Space)
Modify `config.py`:
```python
TRAINING_CONFIG['save_period'] = 5
```

### Disable Early Stopping
Modify `config.py`:
```python
TRAINING_CONFIG['patience'] = 1000  # Very large number
```

## Storage Requirements

### Checkpoint File Sizes
| Model | Size per Checkpoint |
|-------|-------------------|
| yolo11n-seg | ~150 MB |
| yolo11s-seg | ~250 MB |
| yolo11m-seg | ~400 MB |
| yolo11l-seg | ~650 MB |
| yolo11x-seg | ~1 GB |

### Total Storage for 150 Epochs
| Model | Total |
|-------|-------|
| yolo11n-seg | ~22.5 GB |
| yolo11s-seg | ~37.5 GB |
| yolo11m-seg | ~60 GB |
| yolo11l-seg | **~97.5 GB** |
| yolo11x-seg | ~150 GB |

### Reduce Storage
```python
# Save every 5 epochs instead of 1
TRAINING_CONFIG['save_period'] = 5
# Now: 150 epochs ÷ 5 = 30 checkpoints
# yolo11l: 30 × 650 MB ≈ 19.5 GB
```

## Monitoring During Training

### Console Output
```
      Epoch    GPU_mem   box_loss   seg_loss  ...
      1/150      9.94G      2.226      3.383
      2/150      8.85G      2.147      3.244
      3/150      8.75G      2.089      3.156
      ...
     50/150      7.45G      1.234      2.012
      ...
    150/150      7.12G      0.876      1.543

Training complete.
```

### Check Results CSV
```python
import pandas as pd
results = pd.read_csv('xbd_yolo/runs/xbd_seg_v1/results.csv')
print(results[['epoch', 'train/box_loss', 'val/box_loss', 'metrics/mAP50']])
```

### View Training Curves
- Open `results.png` with any image viewer
- Refresh periodically to see updates
- All 8 metrics visualized together

## Advanced: Custom Checkpoint Loading

### Load Specific Epoch
```python
from ultralytics import YOLO

# Load epoch 50
model = YOLO('xbd_yolo/runs/xbd_seg_v1/weights/epoch50.pt')

# Inference
results = model.predict('image.jpg', conf=0.25)
```

### Compare Multiple Epochs
```python
from ultralytics import YOLO
import os

weights_dir = 'xbd_yolo/runs/xbd_seg_v1/weights'
epochs = [1, 10, 50, 100, 150]

for epoch in epochs:
    model = YOLO(f'{weights_dir}/epoch{epoch}.pt')
    # Test on validation set
    metrics = model.val(data='xbd_yolo/xbd.yaml', verbose=False)
    print(f"Epoch {epoch}: mAP50={metrics.seg.map50:.4f}")
```

## Troubleshooting

### No Checkpoints Saved?
✓ Check `save=True` in config  
✓ Check `save_period` is set correctly  
✓ Ensure disk has enough space

### Plots Not Updating?
✓ Check `plots=True` in config  
✓ Plots update at end of epoch, not during  
✓ File is created after epoch 1

### Results CSV Missing?
✓ Created after first epoch completes  
✓ Look in correct run directory

### Out of Disk Space?
✓ Reduce `save_period` (e.g., to 10)  
✓ Use smaller model (yolo11n/s/m)  
✓ Delete old checkpoints after training

## Summary

| Feature | Setting | Value |
|---------|---------|-------|
| Save Checkpoints | `save_period` | **1** (every epoch) |
| Plot Results | `plots` | **True** |
| Early Stop | `patience` | **50** |
| Total Epochs | `epochs` | **150** |

**Result**: Complete training history with all checkpoints and real-time curve updates!
