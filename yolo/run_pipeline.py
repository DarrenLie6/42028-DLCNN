"""Complete YOLO11 training and evaluation pipeline."""

import argparse
import sys
import os
from train import create_dataset_yaml, train_model
from evaluate import (
    load_best_model, run_inference, save_heatmaps,
    aggregate_disaster_heatmap, evaluate_model
)
from config import YOLO_DS_DIR


def run_full_pipeline(train=True, eval=True, num_samples=50):
    """Run complete training and evaluation pipeline."""
    
    print("\n" + "="*70)
    print("YOLO11 xBD Segmentation - Complete Pipeline")
    print("="*70 + "\n")
    
    # Phase 1: Training
    if train:
        print("[1/2] TRAINING PHASE")
        print("-" * 70)
        yaml_path = create_dataset_yaml()
        train_model(yaml_path)
    else:
        yaml_path = os.path.join(YOLO_DS_DIR, "xbd.yaml")
        print(f"[1/2] Skipping training. Using existing model.")
    
    # Phase 2: Evaluation
    if eval:
        print("\n[2/2] EVALUATION PHASE")
        print("-" * 70)
        model = load_best_model()
        pred_results = run_inference(model, split="val", num_samples=num_samples)
        save_heatmaps(pred_results, num_samples=6)
        aggregate_disaster_heatmap(pred_results)
        
        # Validate with official metrics
        if os.path.exists(yaml_path):
            evaluate_model(model, yaml_path)
    else:
        print(f"\n[2/2] Skipping evaluation.")
    
    print("\n" + "="*70)
    print("Pipeline Complete!")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO11 xBD Segmentation Training & Evaluation Pipeline"
    )
    parser.add_argument(
        "--train", 
        action="store_true", 
        default=True,
        help="Run training phase (default: True)"
    )
    parser.add_argument(
        "--no-train",
        dest="train",
        action="store_false",
        help="Skip training phase"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        default=True,
        help="Run evaluation phase (default: True)"
    )
    parser.add_argument(
        "--no-eval",
        dest="eval",
        action="store_false",
        help="Skip evaluation phase"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of validation samples for inference (default: 50)"
    )
    
    args = parser.parse_args()
    
    if not args.train and not args.eval:
        print("Error: At least one of --train or --eval must be specified")
        sys.exit(1)
    
    run_full_pipeline(
        train=args.train,
        eval=args.eval,
        num_samples=args.num_samples
    )


if __name__ == "__main__":
    main()
