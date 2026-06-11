import streamlit as st
import numpy as np
from PIL import Image
import io
import sys
import os
import base64
from pathlib import Path
import rasterio
import tempfile

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from app.inference import DamageAssessor, find_best_checkpoint

st.set_page_config(page_title="ImpactVision", layout="centered")



#   https://raw.githubusercontent.com/<user>/<repo>/<commit>/<path>
BACKGROUND_URL = (
    "https://raw.githubusercontent.com/DarrenLie6/42028-DLCNN/"
    "master/app/assets/2.jpg"
)


def _find_background() -> Path | None:
    assets = ROOT_DIR / "app" / "assets"
    for name in ("background.jpg", "background.jpeg", "background.png"):
        p = assets / name
        if p.exists():
            return p
    return None


def apply_theme(card_opacity: float = 0.92) -> None:
    bg = _find_background()
    if bg is not None:
        # Local file → embed inline (most reliable, works offline)
        ext = bg.suffix.lstrip(".").lower()
        mime = "jpeg" if ext == "jpg" else ext
        b64 = base64.b64encode(bg.read_bytes()).decode()
        image_src = f"data:image/{mime};base64,{b64}"
    elif BACKGROUND_URL:
        # Remote image (must be a direct image URL)
        image_src = BACKGROUND_URL
    else:
        image_src = None

    if image_src is not None:
        background_rule = (
            f'background-image: url("{image_src}");'
            "background-size: cover;"
            "background-position: center;"
            "background-attachment: fixed;"
        )
    else:
        # Fallback gradient so the card still stands out without any image.
        background_rule = "background: linear-gradient(135deg, #1e3a5f 0%, #2c5364 100%);"

    st.markdown(
        f"""
        <style>
        /* Full-page background */
        [data-testid="stAppViewContainer"] {{
            {background_rule}
        }}
        /* Let the background show through the top header bar */
        [data-testid="stHeader"] {{
            background: rgba(0, 0, 0, 0);
        }}
        /* White card around all content (the centred main column) */
        [data-testid="stAppViewContainer"] .block-container {{
            background: rgba(255, 255, 255, {card_opacity});
            padding: 2.5rem 3rem 3rem 3rem;
            border-radius: 18px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.35);
            margin-top: 2.5rem;
            margin-bottom: 2.5rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_theme()

# Checkpoint directories — the app auto-picks the best (highest-mIoU) file in each.
BITEMPORAL_DIR = ROOT_DIR / "checkpoints" / "semantic_seg_transformer"
POSTONLY_DIR   = ROOT_DIR / "checkpoints" / "semantic_seg_transformer_post"


@st.cache_resource
def load_assessor():
    """Build the router once. Models load lazily on first use."""
    bi   = find_best_checkpoint(BITEMPORAL_DIR)
    post = find_best_checkpoint(POSTONLY_DIR)
    if bi is None or post is None:
        st.error(
            f"Missing checkpoints.\n"
            f"  bi-temporal: {bi}\n  post-only: {post}\n"
            f"Train the models first or check the checkpoint directories."
        )
        st.stop()
    return DamageAssessor(bitemporal_ckpt=bi, postonly_ckpt=post)


assessor = load_assessor()

COLOR_MAP = {
    0: [0, 0, 0, 0],          # background (transparent)
    1: [46, 204, 113, 150],   # intact (green)
    2: [241, 196, 15, 150],   # damaged (yellow)
    3: [230, 126, 34, 150],   # destroyed (orange)
}


def load_image_from_bytes(file_bytes, file_name):
    """Load an uploaded image (GeoTIFF or standard format) as a PIL RGB image."""
    file_name_lower = file_name.lower()

    if file_name_lower.endswith((".tif", ".tiff")):
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            with rasterio.open(tmp_path) as src:
                data = src.read([1, 2, 3] if src.count >= 3 else list(range(1, src.count + 1)))
                data = np.transpose(data, (1, 2, 0))  # (H, W, C)

                if data.dtype == np.uint8:
                    img_array = data
                elif data.dtype == np.uint16:
                    img_array = (data / 256).astype(np.uint8)
                else:
                    data_min, data_max = data.min(), data.max()
                    if data_max > data_min:
                        img_array = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
                    else:
                        img_array = (data * 255).astype(np.uint8)

                if img_array.shape[2] == 1:
                    return Image.fromarray(img_array[:, :, 0]).convert("RGB")
                return Image.fromarray(img_array).convert("RGB")
        finally:
            os.unlink(tmp_path)
    else:
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def overlay_mask(post_img: Image.Image, mask: np.ndarray) -> Image.Image:
    """Composite the coloured class mask over the post image."""
    w, h = post_img.size
    rgba_mask = np.zeros((h, w, 4), dtype=np.uint8)
    for class_idx, color in COLOR_MAP.items():
        rgba_mask[mask == class_idx] = color
    mask_img = Image.fromarray(rgba_mask)

    bg = post_img.convert("RGBA")
    if bg.size != (w, h):
        bg = bg.resize((w, h))
    return Image.alpha_composite(bg, mask_img).convert("RGB")


# ── UI ──────────────────────────────────────────────────────────────────────
st.title("ImpactVision: DL Powered Satellite Damage Assessment Tool")
st.write("---")

# Step 1 — pick the model. This drives which uploaders are shown.
st.subheader("Select Model")
model_choice = st.radio(
    "Choose how to assess damage:",
    options=["Pre + Post (change detection)", "Post only"],
    captions=[
        "Higher accuracy — needs both a pre and post event image.",
        "Single image — needs only the post-event image.",
    ],
    horizontal=True,
)
use_bitemporal = model_choice.startswith("Pre + Post")

st.write("---")

# Step 2 — uploaders matched to the selected model.
st.subheader("Upload Satellite Imagery")

if use_bitemporal:
    col_pre, col_post = st.columns(2)
    with col_pre:
        st.write("**Pre-Event Image (RGB) — REQUIRED**")
        pre_file = st.file_uploader(
            "Drag & Drop Pre-Event", type=["png", "jpg", "jpeg", "tif", "tiff"], key="pre"
        )
    with col_post:
        st.write("**Post-Event Image (RGB) — REQUIRED**")
        post_file = st.file_uploader(
            "Drag & Drop Post-Event", type=["png", "jpg", "jpeg", "tif", "tiff"], key="post_bi"
        )
    ready = pre_file is not None and post_file is not None
    missing_msg = "Please upload **both** a pre-event and post-event image."
else:
    pre_file = None
    st.write("**Post-Event Image (RGB) — REQUIRED**")
    post_file = st.file_uploader(
        "Drag & Drop Post-Event", type=["png", "jpg", "jpeg", "tif", "tiff"], key="post_only"
    )
    ready = post_file is not None
    missing_msg = "Please upload a post-event satellite image to begin the assessment."

st.write("---")
st.subheader("Run Assessment")

if ready:
    if st.button("GENERATE COLOUR-GRADED MAP", use_container_width=True):
        with st.spinner("Running model..."):
            post_img = load_image_from_bytes(post_file.getvalue(), post_file.name)
            pre_img = (
                load_image_from_bytes(pre_file.getvalue(), pre_file.name)
                if (use_bitemporal and pre_file is not None) else None
            )

            mask, mode = assessor.predict(post_img, pre_img)
            result_map = overlay_mask(post_img, mask)

            buf = io.BytesIO()
            result_map.save(buf, format="PNG")
            buf.seek(0)

        st.write("---")
        st.subheader("Step 4: Assessment Results")
        st.image(result_map, caption=f"Damage Map Overlay — {mode} model", use_container_width=True)

        st.markdown(
            "**Map Legend:** 🟩 **Intact** &nbsp;&nbsp; 🟨 **Damaged** &nbsp;&nbsp; 🟥 **Destroyed**"
        )

        st.download_button(
            label=" Download Map (.png)",
            data=buf.getvalue(),
            file_name="damage_map.png",
            mime="image/png",
        )
else:
    st.info(missing_msg)
