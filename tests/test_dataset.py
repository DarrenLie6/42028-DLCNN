import numpy as np
from src.data.augmentation_utils import build_train_aug
from src.data.normalization_utils import load_optical, load_sar

def test_optical_sar_alignment():
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        augmentation=SimpleNamespace(
            horizontal_flip_p=1.0,
            vertical_flip_p=0.0,
            rotate_90_p=0.0,
            random_brightness_contrast_p=0.0,
            elastic_transform_p=0.0,
            coarse_dropout_p=0.0,
        )
    )

    aug = build_train_aug(cfg)

    # same pattern
    base = np.arange(16).reshape(4,4).astype(np.float32)
    optical = np.stack([base]*3, axis=-1)
    sar = base[..., None]

    out = aug(image=optical, sar=sar, mask=base)

    # Check they are flipped the same way
    assert np.allclose(out["image"][:,:,0], out["sar"][:,:,0])

def test_missing_files():
    img = load_optical("non_existent.tif", tile_size=128)
    sar = load_sar("non_existent.tif", tile_size=128)

    assert img.shape == (128,128,3)
    assert sar.shape == (128,128,1)
    assert np.all(img == 0)