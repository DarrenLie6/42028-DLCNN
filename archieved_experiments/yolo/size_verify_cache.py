# Run this after build_cache.py to confirm size
import os

cache_dir = r"E:\UTS\CNN and Deep Learning\Assignment 3\42028-DLCNN\xbd_cache"
total = 0
for split in ['train', 'val', 'test']:
    split_path = os.path.join(cache_dir, split)
    files = os.listdir(split_path)
    size  = sum(os.path.getsize(os.path.join(split_path, f)) for f in files)
    total += size
    print(f"{split:6s}: {len(files):5d} files  →  {size/1e9:.2f} GB")
print(f"{'TOTAL':6s}:              →  {total/1e9:.2f} GB")