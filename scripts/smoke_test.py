from omegaconf import OmegaConf
from src.data.dataloader import get_dataloaders

def main():
    cfg = OmegaConf.load("configs/train_config.yaml")
    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    
    # Print dataset sizes
    print("=" * 60)
    print("Dataset Sizes:")
    print("=" * 60)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    if test_loader is not None:
        print(f"Test samples: {len(test_loader.dataset)}")
    else:
        print("Test samples: 0 (no test set)")
    
    print(f"\nBatch sizes:")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    if test_loader is not None:
        print(f"Test batches: {len(test_loader)}")
    print("=" * 60)
    
    batch = next(iter(train_loader))

    print(f"\nBatch shapes:")
    print(f"optical : {batch['optical'].shape}")  # (8, 3, 512, 512)
    print(f"sar     : {batch['sar'].shape}")       # (8, 1, 512, 512)
    print(f"label   : {batch['label'].shape}")     # (8, 512, 512)
    print("Phase 1 complete")

if __name__ == '__main__':
    main()