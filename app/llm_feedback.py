"""
LLM interpretation layer over the damage-segmentation output (via Ollama).

The SegFormer/MiT + DeepLabV3+ model is the *perception* system: it decides,
per pixel, whether a building is Intact / Damaged / Destroyed. This module is
the *interpretation* layer: it takes the model's structured output (the per-class
statistics, and optionally the colour-graded overlay) and asks a local Gemma
model, served by Ollama, to turn it into a concise disaster-response assessment.

Config (optional env vars):
  OLLAMA_MODEL  model tag to use 
  OLLAMA_HOST   server URL (default: http://localhost:11434)
"""

from __future__ import annotations
import base64
import io
import os
from typing import Dict, Iterator, List, Optional

import numpy as np
from PIL import Image
import ollama



MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e4b")
VISION_MAX_SIDE = 768  # downscale the overlay before sending to bound image size

# Class indices produced by the model (see build_transformer_semantic_model).
CLASS_NAMES = {0: "Background", 1: "Intact", 2: "Damaged", 3: "Destroyed"}

SYSTEM_PROMPT = """\
You are a building-damage assessment analyst supporting disaster emergency \
response. You are given the OUTPUT of a semantic-segmentation model \
(a SegFormer/MiT transformer encoder with a DeepLabV3+ decoder) that has \
classified every pixel of post-disaster satellite imagery as Background, \
Intact, Damaged, or Destroyed.

Treat the supplied percentages as ground truth which come from the vision \
model, not from you. Do NOT invent numbers, counts, or locations beyond what \
the statistics (and the image, if provided) support. If an overlay image is \
provided, you may comment on the SPATIAL pattern of damage (clustered vs. \
scattered, edges, concentration) but defer to the percentages for magnitude.

Write a brief, decision-useful assessment with these short sections:
- Severity: one line — an overall severity call (Minimal / Moderate / Severe / \
Catastrophic) with a one-clause justification.
- Key observations: 2-4 bullets on what the distribution implies.
- Recommended response priorities: 2-3 bullets aimed at responders \
(e.g. search-and-rescue, structural triage, access routes).
- Caveats: 1-2 bullets on confidence. If the mode is "post-only", note that \
without a pre-disaster image, intact-vs-pre-existing-condition is harder to \
disambiguate than in bi-temporal change detection.

Be concise and factual. No preamble. Lead with the outcome."""


def compute_damage_stats(mask: np.ndarray) -> Dict[str, float]:
    """
    Reduce an (H, W) class-index mask to the structured stats the LLM reasons over.
    """
    total = int(mask.size)
    building = int((mask > 0).sum())
    denom = building if building > 0 else 1

    counts = {idx: int((mask == idx).sum()) for idx in CLASS_NAMES}
    pct = {
        "intact_pct": counts[1] / denom * 100,
        "damaged_pct": counts[2] / denom * 100,
        "destroyed_pct": counts[3] / denom * 100,
    }
    affected_pct = pct["damaged_pct"] + pct["destroyed_pct"]
    dominant_idx = max((1, 2, 3), key=lambda i: counts[i]) if building else 0

    return {
        "building_coverage_pct": building / total * 100,
        "dominant_class": CLASS_NAMES[dominant_idx],
        "affected_pct": affected_pct,
        **pct,
    }


def _stats_text(stats: Dict[str, float], mode: str) -> str:
    """Format the stats dict as the compact text block sent to the model."""
    return (
        f"Model mode: {mode}\n"
        f"Building footprint coverage (of image): {stats['building_coverage_pct']:.1f}%\n"
        f"Of the building footprint:\n"
        f"  - Intact:    {stats['intact_pct']:.1f}%\n"
        f"  - Damaged:   {stats['damaged_pct']:.1f}%\n"
        f"  - Destroyed: {stats['destroyed_pct']:.1f}%\n"
        f"Affected (damaged + destroyed): {stats['affected_pct']:.1f}%\n"
        f"Dominant building class: {stats['dominant_class']}"
    )


def _image_b64(image: Image.Image) -> str:
    """Downscale a PIL image and return it as a base64 PNG string for Ollama."""
    img = image.convert("RGB")
    img.thumbnail((VISION_MAX_SIDE, VISION_MAX_SIDE), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_messages(
    stats: Dict[str, float], mode: str, image: Optional[Image.Image]
) -> List[dict]:
    user_text = "Assess this building-damage segmentation output.\n\n" + _stats_text(stats, mode)
    if image is not None:
        user_text += (
            "\n\nThe attached image is the colour-graded damage overlay "
            "(green = Intact, yellow = Damaged, red = Destroyed)."
        )
    user_msg: dict = {"role": "user", "content": user_text}
    if image is not None:
        # Ollama accepts a list of base64 strings (vision-capable models only).
        user_msg["images"] = [_image_b64(image)]
    return [{"role": "system", "content": SYSTEM_PROMPT}, user_msg]


def _client():
    if ollama is None:
        raise RuntimeError("The `ollama` package is not installed. Run `pip install ollama`.")
    host = os.environ.get("OLLAMA_HOST")
    return ollama.Client(host=host) if host else ollama.Client()


def stream_damage_feedback(
    stats: Dict[str, float],
    mode: str,
    image: Optional[Image.Image] = None,
) -> Iterator[str]:
    """
    Stream the local model's assessment as text chunks
    """
    client = _client()
    try:
        stream = client.chat(
            model=MODEL,
            messages=_build_messages(stats, mode, image),
            stream=True,
        )
        for chunk in stream:
            piece = chunk["message"]["content"]
            if piece:
                yield piece
    except Exception as e:
        raise RuntimeError(
            f"LLM call failed ({type(e).__name__}: {e}). Check that `ollama serve` "
            f"is running and the model is pulled (`ollama pull {MODEL}`). If it "
            f"crashes only with vision enabled, the model may not support images."
        ) from e


def generate_damage_feedback(
    stats: Dict[str, float],
    mode: str,
    image: Optional[Image.Image] = None,
) -> str:
    """Non-streaming convenience wrapper — returns the full assessment string."""
    return "".join(stream_damage_feedback(stats, mode, image))
