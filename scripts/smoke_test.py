from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.dataloader import get_dataloaders

# Events where pre-event optical is intentionally absent
SAR_ONLY_EVENTS = {"ukraine-conflict", "myanmar-hurricane", "mexico-hurricane"}


def check_missing_files(cfg) -> bool:
    root      = Path(cfg.data.root_dir)
    split_dir = Path(cfg.data.split_file_dir)
    pre_dir   = root / cfg.data.pre_event_dir
    post_dir  = root / cfg.data.post_event_dir
    tgt_dir   = root / cfg.data.target_dir

    all_clean = True

    # Map split names to config keys
    split_mapping = {
        "train": "train_split",
        "val":   "val_split",
        "test":  "test_split"
    }

    for split, config_key in split_mapping.items():
        split_file = split_dir / cfg.data[config_key]
        if not split_file.exists():
            print(f"  WARNING: Split file not found: {split_file}")
            all_clean = False
            continue

        stems = [s.strip() for s in split_file.read_text().splitlines() if s.strip()]

        broken_opt, broken_sar, broken_lbl = [], [], []
        sar_only_count = 0
        event_sar_only = defaultdict(int)

        for stem in stems:
            parts = stem.rsplit("_", 1)
            event = parts[0] if len(parts) == 2 else stem

            opt_missing = not (pre_dir  / f"{stem}_pre_disaster.tif").exists()
            sar_missing = not (post_dir / f"{stem}_post_disaster.tif").exists()
            lbl_missing = not (tgt_dir  / f"{stem}_building_damage.tif").exists()

            if opt_missing:
                if event in SAR_ONLY_EVENTS:
                    sar_only_count += 1
                    event_sar_only[event] += 1
                else:
                    broken_opt.append(stem)

            if sar_missing:
                broken_sar.append(stem)

            if lbl_missing:
                broken_lbl.append(stem)

        n_broken = len(broken_opt) + len(broken_sar) + len(broken_lbl)

        print(f"\n{'='*60}")
        print(f"{split.upper()} SET  ({len(stems)} tiles)")
        print(f"{'='*60}")
        print(f"  [OK] SAR-only tiles (by design) : {sar_only_count:>4}")
        for evt, cnt in sorted(event_sar_only.items()):
            print(f"       {evt:<35}: {cnt}")
        print(f"  [!!] Broken optical             : {len(broken_opt):>4}")
        print(f"  [!!] Broken SAR                 : {len(broken_sar):>4}")
        print(f"  [!!] Broken labels              : {len(broken_lbl):>4}")

        if n_broken == 0:
            print(f"\n  >>> No genuinely broken files - ready!")
        else:
            all_clean = False
            print(f"\n  CRITICAL: {n_broken} broken files found - fix before training!")
            for label, items in [("Missing optical", broken_opt),
                                  ("Missing SAR",     broken_sar),
                                  ("Missing labels",  broken_lbl)]:
                if items:
                    print(f"     {label}:")
                    for s in items[:5]:
                        print(f"       - {s}")
                    if len(items) > 5:
                        print(f"       ... and {len(items)-5} more")

    return all_clean

def check_batch_shapes(loader, cfg):
    batch     = next(iter(loader))
    bs        = cfg.training.batch_size
    tile      = cfg.data.tile_size
    n_valid   = int(batch["optical_valid"].sum().item())
    n_missing = bs - n_valid

    print(f"\nBatch shapes  (batch_size={bs}, tile={tile}):")
    print(f"  optical       : {tuple(batch['optical'].shape)}"
          f"   dtype={batch['optical'].dtype}")
    print(f"  optical_valid : {tuple(batch['optical_valid'].shape)}"
          f"  ->  {n_valid} real  /  {n_missing} SAR-only"
          f"  dtype={batch['optical_valid'].dtype}")
    print(f"  sar           : {tuple(batch['sar'].shape)}"
          f"   dtype={batch['sar'].dtype}")
    print(f"  label         : {tuple(batch['label'].shape)}"
          f"   dtype={batch['label'].dtype}")

    # Verify optical is zeroed where optical_valid=False
    invalid_mask = ~batch["optical_valid"]
    if invalid_mask.any():
        invalid_idx  = invalid_mask.nonzero(as_tuple=True)[0]
        optical_sums = batch["optical"][invalid_idx].abs().sum().item()
        status = "OK" if optical_sums == 0.0 else f"WARNING non-zero sum={optical_sums:.4f}"
        print(f"  optical zeros where invalid  : {status}")

    # Verify label range
    lmin = int(batch["label"].min().item())
    lmax = int(batch["label"].max().item())
    status = "OK" if 0 <= lmin and lmax <= 4 else "WARNING out of range!"
    print(f"  label range   : [{lmin}, {lmax}]  (expected 0-4)  {status}")


def main():
    cfg = OmegaConf.load("configs/train_config.yaml")

    print("\n" + "="*60)
    print("FILE INTEGRITY CHECK")
    print("="*60)
    clean = check_missing_files(cfg)

    print("\n" + "="*60)
    print("DATALOADER CHECK")
    print("="*60)
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Train samples : {len(train_loader.dataset)}")
    print(f"Val   samples : {len(val_loader.dataset)}")
    print(f"Test  samples : {len(test_loader.dataset)}")
    print(f"\nTrain batches : {len(train_loader)}")
    print(f"Val   batches : {len(val_loader)}")
    print(f"Test  batches : {len(test_loader)}")

    print("\n" + "="*60)
    print("BATCH SHAPE VERIFICATION  (train loader)")
    print("="*60)
    check_batch_shapes(train_loader, cfg)

    print("\n" + "="*60)
    if clean:
        print("Phase 1 complete - smoke test passed!")
    else:
        print("FAILED - fix broken files above before training.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()