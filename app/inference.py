"""
Inference router for the transformer DeepLabV3+ damage model.

Routes by what the user uploads:
  * post only        → single-input (post-only) model + post-only weights
  * pre AND post     → bi-temporal (Siamese) model + bi-temporal weights

The mode is auto-detected from each checkpoint (a bi-temporal checkpoint
contains `fusions.*` keys), so the right architecture is rebuilt for whatever
weights file is handed in — no config flag to keep in sync.

Preprocessing matches training exactly: RGB resized to 512×512 and scaled to
[0, 1] (no ImageNet normalisation — the encoder was fine-tuned on [0, 1] inputs).
"""

from __future__ import annotations
import os
import re
import glob
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

from src.models.deeplabv3.deeplabv3plus_transformer import build_transformer_semantic_model


TILE_SIZE = 512
NUM_CLASSES = 4


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def find_best_checkpoint(ckpt_dir: str | Path) -> Optional[Path]:
    """Return the `*.pth` in ckpt_dir with the highest mIoU in its filename."""
    best, best_iou = None, -1.0
    for p in glob.glob(str(Path(ckpt_dir) / "*.pth")):
        m = re.search(r"mIoU_([0-9.]+)", os.path.basename(p))
        if m and float(m.group(1)) > best_iou:
            best_iou, best = float(m.group(1)), Path(p)
    return best


def _is_bitemporal_state_dict(state_dict: dict) -> bool:
    """Bi-temporal checkpoints carry the Siamese fusion blocks."""
    return any(k.startswith("fusions.") for k in state_dict)


def _load_model(
    ckpt_path: str | Path,
    device: torch.device,
    variant: str = "b1",
) -> Tuple[torch.nn.Module, bool]:
    """
    Rebuild the matching architecture for a checkpoint and load its weights.

    Returns (model, bitemporal). `pretrained=True` is used only to reconstruct
    the *same* timm PVTv2 encoder structure the checkpoint was trained with
    (weights are loaded from local timm cache, then overwritten by the
    checkpoint's state_dict).
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    bitemporal = _is_bitemporal_state_dict(state_dict)

    model = build_transformer_semantic_model(
        num_classes=NUM_CLASSES,
        variant=variant,
        aux_loss=True,
        pretrained=True,        # rebuild same encoder structure (uses timm cache)
        bitemporal=bitemporal,
        device=device,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/architecture mismatch for {ckpt_path}: "
            f"{len(missing)} missing, {len(unexpected)} unexpected keys."
        )
    model.eval()
    return model, bitemporal


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def _preprocess(img: Image.Image, size: int = TILE_SIZE) -> torch.Tensor:
    """PIL RGB → (3, size, size) float tensor in [0, 1] (training preprocessing)."""
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
class DamageAssessor:
    """
    Lazy-loading router over the two transformer models.

    Models are loaded on first use (so the app starts fast and only pays for
    the model a given upload actually needs).
    """

    def __init__(
        self,
        bitemporal_ckpt: str | Path,
        postonly_ckpt: str | Path,
        device: Optional[torch.device] = None,
        variant: str = "b1",
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.variant = variant
        self.bitemporal_ckpt = Path(bitemporal_ckpt)
        self.postonly_ckpt = Path(postonly_ckpt)
        self._bi: Optional[torch.nn.Module] = None
        self._post: Optional[torch.nn.Module] = None

    def _bi_model(self) -> torch.nn.Module:
        if self._bi is None:
            model, is_bi = _load_model(self.bitemporal_ckpt, self.device, self.variant)
            if not is_bi:
                raise ValueError(
                    f"{self.bitemporal_ckpt} is not a bi-temporal checkpoint "
                    f"(no fusion layers)."
                )
            self._bi = model
        return self._bi

    def _post_model(self) -> torch.nn.Module:
        if self._post is None:
            model, is_bi = _load_model(self.postonly_ckpt, self.device, self.variant)
            if is_bi:
                raise ValueError(
                    f"{self.postonly_ckpt} is a bi-temporal checkpoint, "
                    f"expected a post-only one."
                )
            self._post = model
        return self._post

    @torch.no_grad()
    def predict(
        self,
        post_img: Image.Image,
        pre_img: Optional[Image.Image] = None,
    ) -> Tuple[np.ndarray, str]:
        """
        Predict a damage mask, routing on whether a pre image was supplied.

        Args:
            post_img: post-disaster RGB image (required)
            pre_img:  pre-disaster RGB image (optional) — if given, the
                      bi-temporal model is used.

        Returns:
            (mask, mode) where mask is an (H, W) uint8 class map at the post
            image's original resolution, and mode is "bi-temporal" or "post-only".
        """
        orig_size = post_img.size  # (W, H)
        post_t = _preprocess(post_img).to(self.device)

        if pre_img is not None:
            pre_t = _preprocess(pre_img).to(self.device)
            x = torch.cat([pre_t, post_t], dim=0).unsqueeze(0)  # (1, 6, H, W)
            model = self._bi_model()
            mode = "bi-temporal"
        else:
            x = post_t.unsqueeze(0)                              # (1, 3, H, W)
            model = self._post_model()
            mode = "post-only"

        logits = model(x)["out"]
        pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)  # (512, 512)

        # Resize the class map back to the original resolution (nearest = no
        # interpolation of class indices).
        pred_full = np.array(
            Image.fromarray(pred).resize(orig_size, Image.NEAREST)
        )
        return pred_full, mode
