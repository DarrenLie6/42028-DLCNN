from pathlib import Path
from omegaconf import OmegaConf
from src.data.dataloader import get_dataloaders

def check_missing_files(cfg):
    """
    Identify missing files required for training and analyze their impact.
    """
    print("\n" + "=" * 80)
    print("FILE INTEGRITY CHECK")
    print("=" * 80)
    
    root_dir = Path(cfg.data.root_dir)
    split_file_dir = Path(cfg.data.split_file_dir)
    
    missing_files = {
        "train": [],
        "val": [],
        "test": []
    }
    
    missing_labels = {
        "train": [],
        "val": [],
        "test": []
    }
    
    # Check all split phases
    split_files = {
        "train": split_file_dir / cfg.data.train_split,
        "val": split_file_dir / cfg.data.val_split,
        "test": split_file_dir / cfg.data.test_split,
    }
    
    total_stems = {}
    
    for phase, split_file in split_files.items():
        if not split_file.exists():
            print(f"\n⚠️  {phase.upper()} split file missing: {split_file}")
            continue
        
        with open(split_file) as f:
            stems = [line.strip() for line in f if line.strip()]
        
        total_stems[phase] = len(stems)
        
        # Check for missing optical, SAR, and label files
        for stem in stems:
            opt_path = root_dir / cfg.data.pre_event_dir / f"{stem}_pre_disaster.tif"
            sar_path = root_dir / cfg.data.post_event_dir / f"{stem}_post_disaster.tif"
            lbl_path = root_dir / cfg.data.target_dir / f"{stem}_building_damage.tif"
            
            if not opt_path.exists():
                missing_files[phase].append((stem, "optical (pre-event)"))
            if not sar_path.exists():
                missing_files[phase].append((stem, "SAR (post-event)"))
            if not lbl_path.exists():
                missing_labels[phase].append(stem)
    
    # Print results
    for phase in ["train", "val", "test"]:
        if phase not in total_stems:
            continue
            
        print(f"\n{phase.upper()} SET:")
        print(f"  Total stems in split file: {total_stems[phase]}")
        
        if missing_files[phase]:
            missing_count = len(set([stem for stem, _ in missing_files[phase]]))
            print(f"  ❌ Missing input files: {missing_count} stems")
            for stem, file_type in sorted(set(missing_files[phase])):
                print(f"     - {stem}: {file_type}")
        else:
            print(f"  ✓ All input files present (optical & SAR)")
        
        if missing_labels[phase]:
            print(f"  ⚠️  Missing label files: {len(missing_labels[phase])} stems")
            for stem in sorted(missing_labels[phase])[:5]:  # Show first 5
                print(f"     - {stem}")
            if len(missing_labels[phase]) > 5:
                print(f"     ... and {len(missing_labels[phase]) - 5} more")
        else:
            print(f"  ✓ All label files present")
    
    # Analyze training impact
    print("\n" + "=" * 80)
    print("TRAINING IMPACT ANALYSIS")
    print("=" * 80)
    
    if missing_files["train"] or missing_files["val"]:
        print("\n❌ CRITICAL: Missing input files will BREAK training!")
        print("   - Optical and SAR images are REQUIRED for both training and validation")
        print("   - Model cannot proceed without these input modalities")
        print("   - Training will crash when processing affected batches")
    else:
        print("\n✓ Input files: No critical issues")
    
    if missing_labels["train"]:
        train_affected = len(missing_labels["train"])
        train_total = total_stems.get("train", 0)
        train_pct = (train_affected / train_total * 100) if train_total > 0 else 0
        print(f"\n⚠️  TRAINING LABELS: {train_affected}/{train_total} stems missing labels ({train_pct:.1f}%)")
        print("   - Missing labels will use zero-filled masks (all 'no damage')")
        print("   - Impact: Reduces training signal; biased towards background class")
        print("   - Recommendation: Regenerate or verify label files before serious training")
    else:
        print("\n✓ Training labels: All present")
    
    if missing_labels["val"]:
        val_affected = len(missing_labels["val"])
        val_total = total_stems.get("val", 0)
        val_pct = (val_affected / val_total * 100) if val_total > 0 else 0
        print(f"\n⚠️  VALIDATION LABELS: {val_affected}/{val_total} stems missing labels ({val_pct:.1f}%)")
        print("   - Validation metrics (IoU, F1, etc.) will be unreliable")
        print("   - Training progress will be hard to assess")
        print("   - Recommendation: Do not rely on validation metrics until fixed")
    else:
        print("\n✓ Validation labels: All present")
    
    if missing_labels["test"] and "test" in total_stems:
        test_affected = len(missing_labels["test"])
        test_total = total_stems.get("test", 0)
        test_pct = (test_affected / test_total * 100) if test_total > 0 else 0
        print(f"\n⚠️  TEST LABELS: {test_affected}/{test_total} stems missing labels ({test_pct:.1f}%)")
        print("   - Test set evaluation will be incomplete")
    else:
        if "test" in total_stems:
            print("\n✓ Test labels: All present (or not critical)")
    
    print("=" * 80 + "\n")


def main():
    cfg = OmegaConf.load("configs/train_config.yaml")
    
    # Phase 0: Check for missing files
    check_missing_files(cfg)
    
    # Phase 1: Load dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    
    # Print dataset sizes
    print("=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    if test_loader is not None:
        print(f"Test samples: {len(test_loader.dataset)}")
    else:
        print("Test samples: 0 (no test set)")
    
    print(f"\nBatch configuration:")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    if test_loader is not None:
        print(f"Test batches: {len(test_loader)}")
    print("=" * 80)
    
    # Phase 2: Verify batch shapes
    batch = next(iter(train_loader))

    print(f"\nBatch shapes (batch_size={cfg.training.batch_size}):")
    print(f"  optical: {batch['optical'].shape}")  # (8, 3, 512, 512)
    print(f"  sar:     {batch['sar'].shape}")       # (8, 1, 512, 512)
    print(f"  label:   {batch['label'].shape}")     # (8, 512, 512)
    print("=" * 80)
    print("✓ Smoke test complete - ready for training!")

if __name__ == '__main__':
    main()