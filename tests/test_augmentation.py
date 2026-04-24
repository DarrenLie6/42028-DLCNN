from types import SimpleNamespace
import numpy as np
from src.data.augmentation_utils import build_train_aug

def test_shapes():
    cfg = SimpleNamespace(
        data=SimpleNamespace(
            root_dir="dummy",
            pre_event_dir="pre",
            post_event_dir="post",
            target_dir="label",
            tile_size=256,
            optical_mean=[0,0,0],
            optical_std=[1,1,1],
            sar_mean=[0],
            sar_std=[1],
        ),
        augmentation=SimpleNamespace(
            horizontal_flip_p=1.0,
            vertical_flip_p=1.0,
            rotate_90_p=1.0,
            random_brightness_contrast_p=1.0,
            elastic_transform_p=1.0,
            coarse_dropout_p=1.0,
        )
    )

    aug = build_train_aug(cfg)

    optical = np.random.rand(256,256,3).astype(np.float32)
    sar = np.random.rand(256,256,1).astype(np.float32)
    label = np.random.randint(0,4,(256,256)).astype(np.int64)

    out = aug(image=optical, sar=sar, mask=label)

    assert out["image"].shape == (256,256,3)
    assert out["sar"].shape == (256,256,1)
    assert out["mask"].shape == (256,256)

def test_augmentation_changes_image():
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

    img = np.random.rand(128,128,3).astype(np.float32)

    out = aug(image=img, sar=img[:,:,0:1], mask=np.zeros((128,128)))

    assert not np.allclose(img, out["image"])