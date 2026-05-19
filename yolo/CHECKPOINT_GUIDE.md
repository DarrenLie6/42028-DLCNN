# Checkpoint & Training Curve Configuration

## What Changed

Updated the YOLO11 training to automatically save checkpoints at each epoch and display training curves.

## Configuration Changes

### In `config.py`:

```python
TRAINING_CONFIG = {
    ...
    'save':          True,        # Save weights (unchanged)
    'save_period':   1,           # ✨ NEW: Save checkpoint every 1 epoch
    'plots':         True,        # Plot results (enhanced)
    'verbose':       True,        # Detailed logging
    'patience':      50,          # Early stopping patience
}
```

**Key additions:**
- `'save_period': 1` - Saves a checkpoint after **every epoch**
- `'patience': 50` - Stops training if validation doesn't improve for 50 epochs

## What Gets Saved During Training

### Checkpoint Files

Every epoch saves a checkpoint to `weights/`:
```
runs/xbd_seg_v1/weights/
├── best.pt       ← Best model (updated when val improves)
├── last.pt       ← Latest epoch
├── epoch1.pt     ← After epoch 1
├── epoch2.pt     ← After epoch 2
├── ...
└── epoch150.pt   ← After epoch 150 (if training completes)
```

### Training Curves (Updated Every Epoch)

After each epoch, YOLO generates plots:
```
runs/xbd_seg_v1/
├── results.csv          ← Metrics per epoch (updated live)
├── results.png          ← Training curves (updated each epoch)
├── confusion_matrix.png
└── labels.jpg
```

## Results CSV File

`results.csv` contains per-epoch metrics:
```
epoch, train/box_loss, train/seg_loss, val/box_loss, val/seg_loss, mAP50, mAP50-95, ...
1,     2.234,         3.102,          2.156,        3.045,        0.234, 0.134,
2,     2.145,         2.987,          2.089,        2.901,        0.312, 0.198,
...
```

## Training Curves Visualization

`results.png` shows 8 plots:
1. **box_loss** - Bounding box prediction error
2. **seg_loss** - Segmentation mask error
3. **cls_loss** - Classification error
4. **dfl_loss** - Distribution focal loss
5. **metrics/precision** - Detection precision
6. **metrics/recall** - Detection recall
7. **val/box_loss** - Validation box loss
8. **val/seg_loss** - Validation segmentation loss

All plots are **updated at the end of each epoch**.

## Resume Training

If training is interrupted, resume from the last checkpoint:

```python
# In Python
from ultralytics import YOLO
model = YOLO('xbd_yolo/runs/xbd_seg_v1/weights/last.pt')
model.train(data='xbd_yolo/xbd.yaml', epochs=200, resume=True)
```

Or via command line:
```bash
yolo detect train data=xbd.yaml model=xbd_seg_v1/weights/last.pt resume=True
```

## Usage

### Train and Save All Checkpoints

```bash
python train.py
```

This will:
1. ✅ Save checkpoint after each epoch to `weights/epoch{N}.pt`
2. ✅ Update `results.csv` after each epoch
3. ✅ Update `results.png` training curves after each epoch
4. ✅ Save best model to `weights/best.pt`
5. ✅ Save last model to `weights/last.pt`

### Monitor Training

Watch the training curves in real-time:
1. Open `results.png` in image viewer
2. Refresh it periodically to see updated curves
3. Check `results.csv` for numerical metrics
4. Monitor console output for live epoch progress

## Early Stopping

With `'patience': 50`, training stops automatically if:
- Validation metrics don't improve for 50 consecutive epochs
- Saves GPU time without sacrificing quality
- Best model is still saved regardless

## Example Training Output

```
      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances   Size
      1/150      9.94G      2.226      3.383      2.042      1.185       1088   640: 100%
      2/150      8.85G      2.147      3.244      1.652      1.158        274   640: 100%
      3/150      8.75G      2.089      3.156      1.512      1.123        456   640: 100%
      ...
     50/150      7.45G      1.234      2.012      0.875      0.623        892   640: 100%
     ...
    150/150      7.12G      0.876      1.543      0.456      0.234        124   640: 100%

Training complete.
Results saved to: xbd_yolo/runs/xbd_seg_v1
✅ Results saved to:
   📁 E:\...\xbd_yolo\runs\xbd_seg_v1

📊 Outputs:
   • best.pt - Best model weights
   • last.pt - Last epoch weights
   • results.csv - Per-epoch metrics
   • results.png - Training curves visualization
   • weights/ - All epoch checkpoints
```

## Checkpoint Storage

With 150 epochs and `save_period=1`:
- **Size per checkpoint**: ~650 MB (for yolo11l-seg)
- **Total for 150 epochs**: ~97.5 GB
- **Keep space**: Ensure sufficient disk space

### Alternative: Save Every N Epochs

To save every 5 epochs instead:
```python
# In config.py
TRAINING_CONFIG['save_period'] = 5
```

This reduces storage to ~20 GB for 150 epochs.

## Monitoring Best Model

Best model is saved automatically when validation improves:
```
best.pt is saved when:
  - Validation loss decreases
  - mAP improves
  - Segmentation metrics improve

Always check results.png to verify training is improving!
```

## Tips

1. **Check results.png regularly** - Loss should decrease, mAP should increase
2. **Save best.pt as backup** - This is your production model
3. **Use last.pt to resume** - If training was interrupted
4. **Archive old checkpoints** - After training, keep only best.pt and last.pt
5. **Monitor GPU memory** - If OOM, reduce batch size or save_period

## Debugging

### No checkpoints saved?
- Check `save_period` is set to 1 or N
- Check `save` is True

### Training curves not updating?
- Check `plots` is True
- Curves update at end of each epoch (not real-time during epoch)

### Results.csv not appearing?
- YOLO creates it after first epoch
- Check correct project/name path

### Out of disk space?
- Reduce `save_period` (e.g., to 10)
- Delete old checkpoints after training completes
- Keep only best.pt for inference

## Summary

| Feature | Before | After |
|---------|--------|-------|
| Checkpoints per epoch | No | ✅ Yes (every epoch) |
| Training curves | Manual | ✅ Automatic (every epoch) |
| Best model tracking | ✅ Yes | ✅ Yes (enhanced) |
| Resume capability | Limited | ✅ Full |
| Early stopping | No | ✅ Yes (50 epochs patience) |

Now training is fully tracked with complete checkpoint history and real-time curve updates!
