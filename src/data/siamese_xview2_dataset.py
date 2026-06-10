"""
Bi-temporal (pre + post) xView2 dataset for the Siamese Attention U-Net.

Returns a 6-channel `image` = [pre(3) | post(3)] so the existing `Trainer`
(which reads `batch["image"]` and calls `model(image)`) works unchanged. The
label is the post-disaster damage mask.

Augmentation note: pre and post must receive the *same* geometric (and colour)
transform, otherwise they de-register. We use albumentations `additional_targets`
so the pre image gets identical parameters to the post image.
"""

from __future__ import annotations
import numpy as np
import cv2
import albumentations as A

from src.data.xview2_dataset import XViewDataset


def build_siamese_train_aug(cfg) -> A.Compose:
    """Training augmentation applied identically to pre and post images."""
    aug = cfg.augmentation
    return A.Compose(
        [
            A.HorizontalFlip(p=aug.horizontal_flip_p),
            A.VerticalFlip(p=aug.vertical_flip_p),
            A.RandomRotate90(p=aug.rotate_90_p),
            A.RandomBrightnessContrast(
                brightness_limit=aug.brightness_contrast.brightness_limit,
                contrast_limit=aug.brightness_contrast.contrast_limit,
                p=aug.brightness_contrast.p,
            ),
            A.GaussNoise(p=aug.gaussian_noise_p),
            A.ElasticTransform(alpha=120, sigma=6, p=aug.elastic_transform_p),
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(8, 32),
                hole_width_range=(8, 32),
                p=aug.coarse_dropout_p,
            ),
        ],
        additional_targets={"pre": "image"},
        is_check_shapes=False,
    )


def build_siamese_val_aug(cfg) -> A.Compose:
    """Validation: no augmentation, but keep additional_targets for API parity."""
    return A.Compose([], additional_targets={"pre": "image"}, is_check_shapes=False)


class SiameseXViewDataset(XViewDataset):
    """
    Bi-temporal variant of XViewDataset.

    Reuses the parent's stem discovery and helpers (_find_image, _load_image,
    _rasterise_label); only __getitem__ changes to also load the pre-disaster
    image and stack pre+post into a 6-channel tensor.
    """

    def __getitem__(self, idx):
        folder, stem = self.stems[idx]

        img_dir = self.root / folder / "images"
        lbl_dir = self.root / folder / "labels"

        post = self._load_image(self._find_image(img_dir, f"{stem}_post_disaster"))
        pre  = self._load_image(self._find_image(img_dir, f"{stem}_pre_disaster"))

        h, w  = post.shape[:2]
        label = self._rasterise_label(lbl_dir / f"{stem}_post_disaster.json", h, w)

        ts = self.tile_size
        if post.shape[:2] != (ts, ts):
            post  = cv2.resize(post,  (ts, ts), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (ts, ts), interpolation=cv2.INTER_NEAREST)
        if pre.shape[:2] != (ts, ts):
            pre   = cv2.resize(pre,   (ts, ts), interpolation=cv2.INTER_LINEAR)

        if self.transform:
            r     = self.transform(image=post, pre=pre, mask=label)
            post  = r["image"]
            pre   = r["pre"]
            label = r["mask"]

        # Stack along channel axis: (H, W, 6) = [pre(3) | post(3)]
        img6 = np.concatenate([pre, post], axis=2)

        import torch
        return {
            "image": torch.from_numpy(np.ascontiguousarray(img6.transpose(2, 0, 1))).float(),
            "label": torch.from_numpy(np.ascontiguousarray(label)).long(),
            "stem":  stem,
        }
