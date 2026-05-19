"""YOLO11 training script for xBD segmentation."""

import os
from ultralytics import YOLO
from config import (
    YOLO_DS_DIR, MODEL_NAME, RUN_NAME, TRAINING_CONFIG, IMG_SIZE, RESUME_CONFIG
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


def get_checkpoint_path(run_name=RUN_NAME, prefer_best=True):
    """
    Find and return the best/last checkpoint path if it exists.
    
    Args:
        run_name: Name of the training run
        prefer_best: If True, prefer best.pt; if False, prefer last.pt
    
    Returns:
        Checkpoint path if found, None otherwise
    """
    runs_dir = os.path.join(YOLO_DS_DIR, "runs")
    weights_dir = os.path.join(runs_dir, run_name, "weights")
    
    if not os.path.exists(weights_dir):
        return None
    
    best_path = os.path.join(weights_dir, "best.pt")
    last_path = os.path.join(weights_dir, "last.pt")
    
    if prefer_best:
        if os.path.exists(best_path):
            return best_path
        elif os.path.exists(last_path):
            return last_path
    else:
        if os.path.exists(last_path):
            return last_path
        elif os.path.exists(best_path):
            return best_path
    
    return None


def train_model(yaml_path, resume_config=RESUME_CONFIG):
    """Train YOLO11 segmentation model with optional checkpoint resuming."""
    
    # Check for existing checkpoint
    checkpoint_path = None
    if resume_config['resume']:
        checkpoint_path = get_checkpoint_path(
            run_name=RUN_NAME,
            prefer_best=resume_config['prefer_best']
        )
    
    print(f"\n{'='*70}")
    print(f"Starting YOLO11 Segmentation Training")
    print(f"{'='*70}")
    
    # Display resume status
    if checkpoint_path:
        checkpoint_name = os.path.basename(checkpoint_path)
        print(f"\n🔄 RESUMING FROM CHECKPOINT")
        print(f"   📁 {checkpoint_path}")
        print(f"   🏆 Using {checkpoint_name}")
    else:
        print(f"\n🆕 STARTING FRESH")
        if resume_config['resume']:
            print(f"   No checkpoint found. Starting from pretrained weights.")
    
    print(f"\n📊 Training Configuration:")
    print(f"  • Model: {MODEL_NAME}")
    print(f"  • Checkpoints: Saved every epoch")
    print(f"  • Training Curves: Updated each epoch")
    print(f"  • Best weights: Saved to best.pt")
    print(f"  • Results location: {os.path.join(YOLO_DS_DIR, 'runs', RUN_NAME)}\n")
    
    # Load model (from checkpoint if available, else pretrained)
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        model = YOLO(checkpoint_path)
    else:
        print(f"Loading pretrained model: {MODEL_NAME}")
        model = YOLO(MODEL_NAME)
    
    # Add project and name to config
    config = TRAINING_CONFIG.copy()
    config['data'] = yaml_path
    config['project'] = os.path.join(YOLO_DS_DIR, "runs")
    config['name'] = RUN_NAME
    
    # Enable resume if loading from checkpoint
    if checkpoint_path:
        config['resume'] = True
        print(f"Resume mode: ENABLED\n")
    
    # Train with checkpoint resuming capability
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
    
    # Train model (with automatic checkpoint resuming)
    train_model(yaml_path, resume_config=RESUME_CONFIG)


if __name__ == "__main__":
    main()
