from __future__ import annotations
import albumentations as A

def build_train_aug(cfg) -> A.Compose:
    """
    Build albumentations composition for training data augmentation.

    cfg : Config
        Configuration object containing augmentation parameters with attributes:
        - horizontal_flip_p : float
            Probability of horizontal flip
        - vertical_flip_p : float
            Probability of vertical flip
        - rotate_90_p : float
            Probability of 90-degree rotation
        - random_brightness_contrast_p : float
            Probability of brightness/contrast adjustment
        - gaussian_noise_p : float
            Probability of Gaussian noise injection
        - elastic_transform_p : float
            Probability of elastic transformation
        - coarse_dropout_p : float
            Probability of coarse dropout

    Returns:
        A.Compose:  Albumentations composition with the configured augmentations.
        Supports SAR (Synthetic Aperture Radar) images as additional targets.
    """
    aug = cfg.augmentation
    return A.Compose([
        A.HorizontalFlip(p=aug.horizontal_flip_p),
        A.VerticalFlip(p=aug.vertical_flip_p),
        A.RandomRotate90(p=aug.rotate_90_p),
        A.RandomBrightnessContrast(p=aug.random_brightness_contrast_p),
        # A.GaussNoise(p=aug.gaussian_noise_p), 
        A.ElasticTransform(alpha=120, sigma=6, p=aug.elastic_transform_p),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            p=aug.coarse_dropout_p,
        ),
    ], additional_targets={"sar": "image"}, is_check_shapes=False)
    
def build_val_aug() -> A.Compose:
    """A wrapper to pass the SAR image type"""
    return  A.Compose([], additional_targets={"sar": "image"})