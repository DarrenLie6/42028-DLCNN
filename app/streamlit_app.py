import streamlit as st
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import io
import sys
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.models.attention_unet import AttentionUNet

st.set_page_config(page_title="ImpactVision", layout="centered")

# load model once at startup
@st.cache_resource
def load_model():
    model = AttentionUNet(num_classes=5)
    weight_path = os.path.join(os.path.dirname(__file__), "../checkpoints/UNet.pth")
    try:
        model.load_state_dict(torch.load(weight_path, map_location="cpu"))
        print("Loaded trained Attention U-Net weights.")
    except FileNotFoundError:
        st.warning("No trained weights found. Using random initialization.")
    model.eval()
    return model

model = load_model()

COLOR_MAP = {
    0: [0, 0, 0, 0],
    1: [46, 204, 113, 150],   # intact
    2: [241, 196, 15, 150],   # minor damage
    3: [230, 126, 34, 150],   # major damage
    4: [231, 76, 60, 150]     # destroyed
}

def run_inference(post_img, pre_img=None):
    opt_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
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

    # overlay on pre or post
    bg = pre_img.convert("RGBA") if pre_img else post_img.convert("RGBA")
    if bg.size != original_size:
        bg = bg.resize(original_size)

    overlay = Image.alpha_composite(bg, mask_img).convert("RGB")
    return overlay

# UI
st.title("ImpactVision: Satellite Damage Assessment Tool")
st.write("---")

st.subheader("Step 1: Upload Satellite Imagery")
col1, col2 = st.columns(2)

with col1:
    st.write("**Post-Event Image (Optical RGB) - REQUIRED**")
    post_file = st.file_uploader("Drag & Drop Post-Event", type=["png", "jpg", "jpeg", "tif"])

with col2:
    st.write("**Pre-Event Image (Optical RGB) - OPTIONAL**")
    pre_file = st.file_uploader("Drag & Drop Pre-Event", type=["png", "jpg", "jpeg", "tif"])

st.write("---")
st.subheader("Step 2: Run Assessment")

if post_file is not None:
    if st.button("🚀 GENERATE COLOUR-GRADED MAP", use_container_width=True):
        with st.spinner("Running model..."):
            post_img = Image.open(io.BytesIO(post_file.getvalue())).convert("RGB")
            pre_img  = Image.open(io.BytesIO(pre_file.getvalue())).convert("RGB") if pre_file else None

            result_map = run_inference(post_img, pre_img)

            buf = io.BytesIO()
            result_map.save(buf, format="PNG")
            buf.seek(0)

        st.write("---")
        st.subheader("Step 3: Assessment Results")
        caption_text = "Damage Map Overlay (on Pre-Event)" if pre_file else "Damage Map Overlay (on Post-Event)"
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
    st.info("Please upload a Post-Event optical image to unlock the assessment button.")