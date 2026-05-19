"""YOLO11 training script for xBD segmentation."""

import os
from ultralytics import YOLO
from config import (
    YOLO_DS_DIR, MODEL_NAME, RUN_NAME, TRAINING_CONFIG, IMG_SIZE
)


def create_dataset_yaml():
    """Create dataset YAML file for YOLO."""
    yaml_content = f"""path: {YOLO_DS_DIR.replace(chr(92), '/')}
train: images/train
val:   images/val
test:  images/test

nc: 3
names: ['intact', 'damaged', 'destroyed']
"""
    
    yaml_path = os.path.join(YOLO_DS_DIR, "xbd.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content.strip())
    
    print(f"Dataset YAML written → {yaml_path}\n{yaml_content}")
    return yaml_path


def train_model(yaml_path):
    """Train YOLO11 segmentation model with checkpoint saving at each epoch."""
    print(f"\n{'='*70}")
    print(f"Starting YOLO11 Segmentation Training")
    print(f"{'='*70}")
    print(f"\n📊 Training Configuration:")
    print(f"  • Model: {MODEL_NAME}")
    print(f"  • Checkpoints: Saved every epoch")
    print(f"  • Training Curves: Updated each epoch")
    print(f"  • Best weights: Saved to best.pt")
    print(f"  • Results location: {os.path.join(YOLO_DS_DIR, 'runs', RUN_NAME)}\n")
    
    # Load YOLO11 segmentation model
    model = YOLO(MODEL_NAME)
    
    # Add project and name to config
    config = TRAINING_CONFIG.copy()
    config['data'] = yaml_path
    config['project'] = os.path.join(YOLO_DS_DIR, "runs")
    config['name'] = RUN_NAME
    
    # Train with checkpoint saving
    print(f"Starting training loop...\n")
    results = model.train(**config)
    
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"\n✅ Results saved to:")
    run_dir = os.path.join(config['project'], config['name'])
    print(f"   📁 {run_dir}")
    print(f"\n📊 Outputs:")
    print(f"   • best.pt - Best model weights")
    print(f"   • last.pt - Last epoch weights")
    print(f"   • results.csv - Per-epoch metrics")
    print(f"   • results.png - Training curves visualization")
    print(f"   • weights/ - All epoch checkpoints (epoch1.pt, epoch2.pt, ...)")
    print(f"\n{'='*70}\n")
    
    return results


def main():
    """Main training pipeline."""
    # Create dataset YAML
    yaml_path = create_dataset_yaml()
    
    # Train model
    train_model(yaml_path)


if __name__ == "__main__":
    main()
