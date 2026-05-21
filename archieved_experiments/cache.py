"""
build_cache.py  —  Run LOCALLY before uploading to SageMaker.
Converts raw xBD GeoTIFFs → compact SAM2 feature tensors + masks.
Output size: ~15-20 GB (vs 180 GB raw)
"""

import os, glob, json, torch, tifffile, numpy as np
from PIL import Image
from shapely import wkt
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
from sam2.build_sam import build_sam2
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DATA_DIR  = r"E:\UTS\CNN and Deep Learning\Assignment 3\42028-DLCNN\data\xView2\geotiffs"
CACHE_OUT_DIR  = r"D:\xbd_cache"
SAM2_CFG       = "sam2_hiera_l.yaml"
SAM2_CKPT      = r"E:\UTS\CNN and Deep Learning\Assignment 3\42028-DLCNN\sam2\sam2\checkpoints\sam2_hiera_large.pt"
IMG_SIZE       = 512
BATCH_SIZE     = 4
SAVE_DTYPE     = torch.float16   # halves storage vs float32

SPLITS = {
    "train": [os.path.join(BASE_DATA_DIR, "tier1"),
              os.path.join(BASE_DATA_DIR, "tier3")],
    "val":   [os.path.join(BASE_DATA_DIR, "hold")],
    "test":  [os.path.join(BASE_DATA_DIR, "test")],
}

XBD_TO_CLASS = {
    'no-damage': 1, 'minor-damage': 2,
    'major-damage': 2, 'destroyed': 3,
}

img_transform = T.Compose([
    T.ToTensor(),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])

# ── Mask builder ──────────────────────────────────────────────────────────────
import cv2

def build_seg_mask(buildings, image_shape=(1024, 1024)):
    mask = np.zeros(image_shape, dtype=np.int32)
    draw_order = [
        ('no-damage', 1), ('minor-damage', 2),
        ('major-damage', 2), ('destroyed', 3),
    ]
    for damage_type, class_idx in draw_order:
        for b in buildings:
            if b['damage'] == damage_type:
                pts    = b['polygon'].reshape((-1, 1, 2)).astype(np.int32)
                mask_c = np.ascontiguousarray(mask)
                cv2.fillPoly(mask_c, [pts], class_idx)
                mask   = mask_c
    return mask.astype(np.int64)


# ── Inline Dataset (no worker issues) ────────────────────────────────────────
class XBDRawDataset(Dataset):
    def __init__(self, dirs):
        self.samples = []
        for d in (dirs if isinstance(dirs, list) else [dirs]):
            for lp in glob.glob(os.path.join(d, 'labels', '*post_disaster.json')):
                base = os.path.basename(lp).replace('_post_disaster.json', '')
                pre  = os.path.join(d, 'images', f'{base}_pre_disaster.tif')
                post = os.path.join(d, 'images', f'{base}_post_disaster.tif')
                if os.path.exists(pre) and os.path.exists(post):
                    self.samples.append((pre, post, lp))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        pre_p, post_p, lbl_p = self.samples[idx]

        pre_img  = np.array(tifffile.imread(pre_p))
        post_img = np.array(tifffile.imread(post_p))
        if pre_img.ndim  == 2: pre_img  = np.stack([pre_img]  * 3, -1)
        if post_img.ndim == 2: post_img = np.stack([post_img] * 3, -1)
        pre_img  = pre_img.astype(np.uint8)  if pre_img.max()  > 1 else (pre_img  * 255).astype(np.uint8)
        post_img = post_img.astype(np.uint8) if post_img.max() > 1 else (post_img * 255).astype(np.uint8)

        with open(lbl_p) as f:
            label = json.load(f)
        buildings = []
        for feat in label['features']['xy']:
            damage = feat['properties'].get('subtype', 'no-damage')
            poly   = wkt.loads(feat['wkt'])
            coords = np.array(poly.exterior.coords, dtype=np.int32)
            buildings.append({'polygon': coords, 'damage': damage})

        seg_mask = build_seg_mask(buildings,
                                  image_shape=(pre_img.shape[0], pre_img.shape[1]))

        pre_t  = img_transform(Image.fromarray(pre_img))
        post_t = img_transform(Image.fromarray(post_img))
        mask_t = torch.from_numpy(
            np.array(Image.fromarray(seg_mask.astype(np.int16))
                     .resize((IMG_SIZE, IMG_SIZE), Image.NEAREST))
        ).long()

        return pre_t, post_t, mask_t


# ── Load SAM2 encoder once ────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

sam2 = build_sam2(SAM2_CFG, SAM2_CKPT, device=device)
encoder = sam2.image_encoder.eval()
for p in encoder.parameters():
    p.requires_grad = False

def encode(imgs):
    with torch.no_grad():
        out = encoder(imgs.to(device))
    # ✅ Use index [0] (32×32) instead of [-1] (64×64) — 4× smaller feature maps
    return out['backbone_fpn'][0].cpu().to(SAVE_DTYPE)  # (B,256,32,32) float16


# ── Build cache split by split ────────────────────────────────────────────────
total_saved = 0

for split_name, dirs in SPLITS.items():
    split_dir = os.path.join(CACHE_OUT_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)

    dataset = XBDRawDataset(dirs)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=0)

    print(f"\n[{split_name}] {len(dataset)} scenes → {split_dir}")

    global_idx = 0
    for pre, post, masks in tqdm(loader, desc=split_name):
        pre_feats  = encode(pre)    # (B,256,h,w) float16
        post_feats = encode(post)   # (B,256,h,w) float16

        for i in range(pre.size(0)):
            out_path = os.path.join(split_dir, f"{global_idx:06d}.pt")
            torch.save({
                'pre_feat':  pre_feats[i],   # (256,h,w) float16
                'post_feat': post_feats[i],  # (256,h,w) float16
                'mask':      masks[i],       # (H,W)     int64
            }, out_path)
            global_idx += 1
            total_saved += 1

    # Print split size
    split_bytes = sum(
        os.path.getsize(os.path.join(split_dir, f))
        for f in os.listdir(split_dir)
    )
    print(f"  → {global_idx} files, {split_bytes/1e9:.2f} GB")

print(f"\nTotal cached: {total_saved} files")
print(f"Total cache size: "
      f"{sum(os.path.getsize(os.path.join(r,f)) for r,d,files in os.walk(CACHE_OUT_DIR) for f in files)/1e9:.2f} GB")