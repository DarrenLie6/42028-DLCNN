import streamlit as st
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import io
import sys
import os
from pathlib import Path
import rasterio
import tempfile

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.models.attention_unet import AttentionUNet

st.set_page_config(page_title="ImpactVision", layout="centered")

# load model once at startup
@st.cache_resource
def load_model():
    model = AttentionUNet(num_classes=4)  # xView2: 4 classes (0=bg, 1=intact, 2=damaged, 3=destroyed)
    weight_path = ROOT_DIR / "checkpoints" / "xview2" / "UNet.pth"
    try:
        checkpoint = torch.load(weight_path, map_location="cpu")
        # Extract model state dict if checkpoint has wrapped metadata
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            # Direct state dict (no metadata wrapper)
            model.load_state_dict(checkpoint)
        print("Loaded trained Attention U-Net weights.")
    except FileNotFoundError:
        st.warning("No trained weights found. Using random initialization.")
    except Exception as e:
        st.warning(f"Error loading weights: {e}. Using random initialization.")
    model.eval()
    return model

model = load_model()

COLOR_MAP = {
    0: [0, 0, 0, 0],          # background (transparent)
    1: [46, 204, 113, 150],   # intact (green)
    2: [241, 196, 15, 150],   # damaged (yellow)
    3: [230, 126, 34, 150]    # destroyed (orange)
}

def load_image_from_bytes(file_bytes, file_name):
    """
    Load image from bytes, supporting both GeoTIFF and standard formats.
    
    Args:
        file_bytes: bytes from file upload
        file_name: original file name (to detect format)
    
    Returns:
        PIL Image (RGB)
    """
    # Detect file type from extension
    file_name_lower = file_name.lower()
    
    if file_name_lower.endswith(('.tif', '.tiff')):
        # Use rasterio for GeoTIFF files
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        
        try:
            with rasterio.open(tmp_path) as src:
                # Read first 3 bands (RGB)
                data = src.read([1, 2, 3] if src.count >= 3 else list(range(1, src.count + 1)))
                data = np.transpose(data, (1, 2, 0))  # (H, W, C)
                
                # Normalize to [0, 255]
                if data.dtype == np.uint8:
                    img_array = data
                elif data.dtype == np.uint16:
                    img_array = (data / 256).astype(np.uint8)
                else:
                    # Float data - normalize to [0, 255]
                    data_min = data.min()
                    data_max = data.max()
                    if data_max > data_min:
                        img_array = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
                    else:
                        img_array = (data * 255).astype(np.uint8)
                
                # Convert to PIL Image
                if img_array.shape[2] == 1:
                    img = Image.fromarray(img_array[:, :, 0]).convert("RGB")
                else:
                    img = Image.fromarray(img_array).convert("RGB")
                return img
        finally:
            os.unlink(tmp_path)
    else:
        # Use PIL for standard formats (PNG, JPG, etc.)
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return img

def run_inference(post_img):
    opt_transform = transforms.Compose([
        transforms.Resize((512, 512)),  # xView2 training size
        transforms.ToTensor(),
        # xView2 images are already normalized to [0,1] by loader
        # No additional normalization needed
    ])
    
    original_size = post_img.size
    post_tensor = opt_transform(post_img).unsqueeze(0)

    with torch.no_grad():
        logits = model(post_tensor)

    preds = torch.argmax(logits, dim=1).squeeze(0).numpy()
    preds_img = Image.fromarray(preds.astype(np.uint8)).resize(original_size, Image.NEAREST)
    preds_resized = np.array(preds_img)

    # color mask
    rgba_mask = np.zeros((original_size[1], original_size[0], 4), dtype=np.uint8)
    for class_idx, color in COLOR_MAP.items():
        rgba_mask[preds_resized == class_idx] = color
    mask_img = Image.fromarray(rgba_mask)

    # overlay on post-event image
    bg = post_img.convert("RGBA")
    if bg.size != original_size:
        bg = bg.resize(original_size)

    overlay = Image.alpha_composite(bg, mask_img).convert("RGB")
    return overlay

# UI
st.title("ImpactVision: AI Powered Satellite Damage Assessment Tool")
st.write("---")

st.subheader("Step 1: Upload Satellite Imagery")
st.write("**Post-Event Image (Optical RGB) - REQUIRED**")
post_file = st.file_uploader("Drag & Drop Post-Event", type=["png", "jpg", "jpeg", "tif", "tiff"])

st.write("---")
st.subheader("Step 2: Run Assessment")

if post_file is not None:
    if st.button("🚀 GENERATE COLOUR-GRADED MAP", use_container_width=True):
        with st.spinner("Running model..."):
            post_img = load_image_from_bytes(post_file.getvalue(), post_file.name)

            result_map = run_inference(post_img)

            buf = io.BytesIO()
            result_map.save(buf, format="PNG")
            buf.seek(0)

        st.write("---")
        st.subheader("Step 3: Assessment Results")
        caption_text = "Damage Map Overlay"
        st.image(result_map, caption=caption_text, use_container_width=True)

        st.markdown("""
        **Map Legend:** 🟩 **Intact** &nbsp;&nbsp; 🟨 **Minor Damage** &nbsp;&nbsp; 🟧 **Major Damage** &nbsp;&nbsp; 🟥 **Destroyed**
        """)

        st.download_button(
            label="📥 Download Map (.png)",
            data=buf.getvalue(),
            file_name="damage_map.png",
            mime="image/png"
        )
else:
    st.info("Please upload a satellite image to begin the assessment.")