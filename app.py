import streamlit as st
import numpy as np
from PIL import Image
import io

# --- 1. DUMMY U-NET MODEL ---
# You will replace this function with your actual PyTorch/Keras inference script later.
def run_unet_inference(optical_img_file, sar_img_file):
    # Open the uploaded optical image to get its width and height
    img = Image.open(optical_img_file)
    width, height = img.size
    
    # Simulate a U-Net output mask by generating random classes (0, 1, 2, 3)
    # 0: Background, 1: Intact, 2: Damaged, 3: Destroyed
    # We heavily weight '0' to simulate background dominance in satellite imagery
    random_classes = np.random.choice(
        [0, 1, 2, 3], 
        size=(height, width), 
        p=[0.85, 0.08, 0.05, 0.02]
    )
    
    # Map the numerical classes to RGB colours for the map
    color_map = {
        0: [0, 0, 0],          # Black (Background)
        1: [0, 255, 0],        # Green (Intact)
        2: [255, 165, 0],      # Orange (Damaged)
        3: [255, 0, 0]         # Red (Destroyed)
    }
    
    # Create an empty RGB array and fill it with the colours
    rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for class_idx, color in color_map.items():
        rgb_mask[random_classes == class_idx] = color
        
    return Image.fromarray(rgb_mask)


# --- 2. STREAMLIT GUI DESIGN ---
st.set_page_config(page_title="ImpactVision", layout="centered")

st.title("ImpactVision: Satellite Damage Assessment Tool")
st.write("---")

# Step 1: Uploads
st.subheader("Step 1: Upload Satellite Imagery")
col1, col2 = st.columns(2)

with col1:
    st.write("**Pre-Event Image (Optical RGB)**")
    opt_file = st.file_uploader("Drag & Drop Optical", type=["png", "jpg", "jpeg", "tif"])

with col2:
    st.write("**Post-Event Image (SAR Greyscale)**")
    sar_file = st.file_uploader("Drag & Drop SAR", type=["png", "jpg", "jpeg", "tif"])

st.write("---")

# Step 2: Assessment Trigger
st.subheader("Step 2: Run Assessment")

# Ensure the user has uploaded both files before allowing them to click the button
if opt_file is not None and sar_file is not None:
    # A large, full-width button
    if st.button("🚀 GENERATE COLOUR-GRADED MAP", use_container_width=True):
        
        with st.spinner("Running Siamese U-Net Inference (Simulated)..."):
            # Call our dummy model
            result_map = run_unet_inference(opt_file, sar_file)
            
            st.write("---")
            
            # Step 3: Output
            st.subheader("Step 3: Assessment Results")
            
            # Display the resulting map
            st.image(result_map, caption="Colour-Graded Damage Map (U-Net Output)", use_column_width=True)
            
            # Display the Legend
            st.markdown("""
            **Map Legend:** 🟩 **Intact** &nbsp;&nbsp;&nbsp; 🟧 **Damaged** &nbsp;&nbsp;&nbsp; 🟥 **Destroyed**
            """)
            
            # Convert the PIL image to bytes so the user can download it
            buf = io.BytesIO()
            result_map.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            # Download Button
            st.download_button(
                label="📥 Download Map (.png)",
                data=byte_im,
                file_name="unet_damage_map.png",
                mime="image/png"
            )
else:
    st.info("Please upload both Pre-Event and Post-Event images to unlock the assessment button.")