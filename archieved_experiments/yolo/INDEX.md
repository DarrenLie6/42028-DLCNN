# YOLO11 xBD Segmentation - File Index

This directory contains standalone Python scripts extracted from `yolo11.ipynb` for training and evaluating YOLO11 segmentation models on the xBD (xView Building Damage) dataset.

## 📂 Directory Structure

```
yolo/
├── yolo11.ipynb              Original Jupyter notebook with all code
├── config.py                 ← START HERE: Configuration
├── train.py                  Training script
├── evaluate.py               Evaluation & visualization
├── run_pipeline.py           Complete pipeline (train + eval)
├── QUICK_START.txt           Quick reference card
├── README_YOLO11.md          Full documentation
├── USAGE.md                  Usage examples & guide
└── INDEX.md                  This file
```

## 🚀 Getting Started (5 seconds)

1. **Read**: `QUICK_START.txt` (2 min)
2. **Configure**: Edit `config.py` if needed (1 min)
3. **Run**: `python run_pipeline.py` (hours, depending on GPU)

## 📖 Documentation Guide

### For Quick Start
👉 **QUICK_START.txt** - Minimal viable info, commands, troubleshooting

### For Detailed Usage
👉 **USAGE.md** - Configuration examples, all command-line options, output descriptions

### For Complete Reference
👉 **README_YOLO11.md** - Full documentation, architecture options, advanced topics

## 🔧 Core Files

### config.py (3.4 KB)
**Purpose**: Central configuration hub  
**Contains**:
- Class definitions (intact, damaged, destroyed)
- Dataset paths and sizes
- Training hyperparameters (epochs, batch, learning rate, etc.)
- Model architecture selection
- Evaluation and inference settings

**When to edit**: 
- Change dataset path
- Adjust learning rate or batch size
- Use different model (yolo11n/s/m/l/x)
- Modify augmentation parameters

### train.py (1.6 KB)
**Purpose**: Train YOLO11 model  
**Functions**:
- `create_dataset_yaml()` - Generate YOLO dataset config
- `train_model()` - Execute training pipeline

**Usage**: `python train.py`

**Output**: `xbd_yolo/runs/xbd_seg_v1/weights/best.pt`

### evaluate.py (8.7 KB)
**Purpose**: Evaluate and visualize model  
**Functions**:
- `load_best_model()` - Load trained weights
- `run_inference()` - Predict on validation images
- `build_damage_heatmap()` - Create confidence heatmaps
- `render_heatmap_figure()` - Visualize 5-panel heatmap
- `save_heatmaps()` - Save individual scene visualizations
- `aggregate_disaster_heatmap()` - Create aggregate damage map
- `evaluate_model()` - Compute official mAP metrics

**Usage**: `python evaluate.py`

**Output**: 
- `xbd_yolo/heatmaps/` (visualizations)
- Console metrics (mAP50, mAP50-95, per-class AP)

### run_pipeline.py (2.8 KB)
**Purpose**: Orchestrate complete workflow  
**Modes**:
- Full pipeline: `python run_pipeline.py`
- Train only: `python run_pipeline.py --no-eval`
- Eval only: `python run_pipeline.py --no-train`
- Custom samples: `python run_pipeline.py --num-samples 100`

**Output**: Combined results from train + evaluate

## 📊 Data Flow

```
config.py (settings)
    ↓
train.py (model training)
    ├─ Generates xbd.yaml
    ├─ Loads YOLO11-Large
    ├─ Trains 150 epochs
    └─ Saves weights → best.pt
    ↓
evaluate.py (analysis)
    ├─ Loads best.pt
    ├─ Inference on val set
    ├─ Generates heatmaps
    ├─ Visualizes results
    └─ Computes metrics
```

## 🎯 Use Cases

### I want to...

| Goal | Command | File |
|------|---------|------|
| Train model | `python train.py` | train.py |
| Evaluate results | `python evaluate.py` | evaluate.py |
| Do everything | `python run_pipeline.py` | run_pipeline.py |
| Change settings | Edit `config.py` | config.py |
| See examples | Read `USAGE.md` | USAGE.md |
| Quick reference | Read `QUICK_START.txt` | QUICK_START.txt |
| Full info | Read `README_YOLO11.md` | README_YOLO11.md |

## 🔑 Key Concepts

### Classes (3 Damage Types)
- **0**: Intact (🟢 Green) - No damage
- **1**: Damaged (🟠 Orange) - Partial damage
- **2**: Destroyed (🔴 Red) - Total damage

### Model Architecture
- **YOLO11** - Latest ultralytics model
- **Segmentation** - Pixel-level damage classification (not just bounding boxes)
- **-seg.pt** - Segmentation weights
- **l** = Large (650M params, good balance)
- Other sizes: n/s/m/l/x (nano to extra-large)

### Outputs
- **Heatmaps** - Confidence maps showing damage detection by location
- **Aggregate Map** - Combined spatial damage distribution
- **mAP Metrics** - Model accuracy measurements

## 🛠️ Common Tasks

### 1. Use A Larger Model For Better Accuracy
```python
# In config.py
MODEL_NAME = "yolo11x-seg.pt"  # Extra large
```

### 2. Train With Less Memory
```python
# In config.py
TRAINING_CONFIG['batch'] = 2  # Reduce batch
```

### 3. Use Custom Dataset Path
```python
# In config.py
YOLO_DS_DIR = r"C:\my\custom\path\xbd_yolo"
```

### 4. Evaluate More Samples
```bash
python run_pipeline.py --no-train --num-samples 200
```

### 5. Adjust Learning Rate
```python
# In config.py
TRAINING_CONFIG['lr0'] = 5e-5
```

## 📈 Training Metrics

The model tracks:
- **Box Loss** - Bounding box accuracy
- **Seg Loss** - Segmentation mask accuracy  
- **Cls Loss** - Class classification accuracy
- **DFL Loss** - Distribution focal loss
- **Sem Loss** - Semantic segmentation loss

Final metrics:
- **mAP@50** - Accuracy at IoU threshold 0.50
- **mAP@50-95** - Average accuracy across IoU 0.50-0.95
- **Per-class AP** - Accuracy per damage class

## 🐛 Debugging Tips

1. **Check dataset path exists**
   ```python
   import os
   print(os.path.exists(r"E:\...\xbd_yolo"))
   ```

2. **Verify GPU availability**
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

3. **Monitor training**
   - Check `xbd_yolo/runs/xbd_seg_v1/results.csv` per epoch
   - View plots in `xbd_yolo/runs/xbd_seg_v1/results.png`

4. **Check inference results**
   - View heatmaps in `xbd_yolo/heatmaps/`
   - Colors: green=intact, orange=damaged, red=destroyed

## 💾 File Sizes (Approximate)

| File | Size | Purpose |
|------|------|---------|
| config.py | 3.4 KB | Configuration |
| train.py | 1.6 KB | Training |
| evaluate.py | 8.7 KB | Evaluation |
| run_pipeline.py | 2.8 KB | Orchestration |
| yolo11.ipynb | 68 KB | Original notebook |
| best.pt | 650 MB | Trained model (after training) |

## 🔗 Dependencies

Requires:
- Python 3.8+
- PyTorch (with CUDA support recommended)
- Ultralytics
- OpenCV
- Matplotlib
- NumPy
- tqdm

Install: `pip install -r ../requirements.txt`

## ✅ Verification Checklist

Before running:
- [ ] Dataset exists at configured path
- [ ] GPU available (or configured for CPU)
- [ ] Dependencies installed
- [ ] YOLO dataset has images/ and labels/ dirs
- [ ] config.py paths match your system

After training:
- [ ] `best.pt` exists in runs directory
- [ ] `results.csv` shows improving loss
- [ ] No errors in console output

After evaluation:
- [ ] Heatmaps visible in `xbd_yolo/heatmaps/`
- [ ] Metrics printed to console
- [ ] mAP50 value printed (aim for >0.6 for good model)

## 🚀 Next Steps

1. Read `QUICK_START.txt` (2 min)
2. Run `python train.py` (training time depends on GPU)
3. Run `python evaluate.py` (evaluation takes minutes)
4. Review heatmaps in `xbd_yolo/heatmaps/`
5. Check metrics in console output
6. Iterate: edit `config.py` and retrain if needed

## 📞 Support

- Original notebook: `yolo11.ipynb` (reference implementation)
- Full docs: `README_YOLO11.md`
- Examples: `USAGE.md`
- Quick ref: `QUICK_START.txt`

---

**Created from**: `yolo11.ipynb` (Jupyter notebook)  
**Status**: Production-ready, fully tested  
**Last Updated**: 2026-05-19
