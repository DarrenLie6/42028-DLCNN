import streamlit as st
import numpy as np
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
# from models.siamese_unet import SiameseUNet

# --- 1. DUMMY U-NET MODEL ---
# You will replace this function with your actual PyTorch/Keras inference script later.
def run_unet_inference(optical_img_file, sar_img_file):
    # Open the uploaded optical image to get its width and height
    img = Image.open(optical_img_file)
    width, height = img.size
    
    # Simulate a U-Net output mask by generating random classes (0, 1, 2, 3)
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

# ---Model Loading---
# @st.cache_resource
# def load_model():
#     # Initialize the architecture with 5 classes to match your EDA findings
#     # (0: Background, 1: Intact, 2: Minor, 3: Major, 4: Destroyed)
#     model = SiameseUNet(num_classes=5, pretrained=False)
    
#     # ⚠️ INSTRUCTION FOR GROUP MEMBER ⚠️
#     # Once the model is trained, place the .pth file in a 'weights' folder
#     # and uncomment the following two lines:
#     #
#     # weight_path = "weights/best_unet_model.pth"
#     # model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    
#     model.eval() # Set to evaluation mode
#     return model

# # --- 2. INFERENCE SCRIPT ---
# def run_unet_inference(optical_img_file, sar_img_file):
#     model = load_model()

#     # 1. Load images
#     opt_img_raw = Image.open(optical_img_file).convert("RGB")
#     sar_img_raw = Image.open(sar_img_file).convert("L") # 'L' ensures 1-channel grayscale
#     original_size = opt_img_raw.size # (Width, Height)

#     # 2. Preprocessing (Resize to 256x256 and convert to tensor)
#     # Using standard ImageNet normalization for optical
#     opt_transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     sar_transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor()
#     ])

#     # 3. Apply transforms and add batch dimension (B, C, H, W)
#     opt_tensor = opt_transform(opt_img_raw).unsqueeze(0)
#     sar_tensor = sar_transform(sar_img_raw).unsqueeze(0)

#     opt_valid = torch.tensor([1.0])

#     with torch.no_grad():
#         logits = model(opt_tensor, sar_tensor, opt_valid)
        
#     preds = torch.argmax(logits, dim=1).squeeze(0).numpy() # Shape: (256, 256)

#     # Convert prediction array back to an image and resize it to original upload resolution
#     preds_img = Image.fromarray(preds.astype(np.uint8)).resize(original_size, Image.NEAREST)
#     preds_resized = np.array(preds_img)

#     # 6. Apply RGBA Color Map to predictions (Matching your EDA hex colors)
#     color_map = {
#         0: [0, 0, 0, 0],         # Background 
#         1: [46, 204, 113, 150],  # Intact
#         2: [241, 196, 15, 150],  # Minor
#         3: [230, 126, 34, 150],  # Major
#         4: [231, 76, 60, 150]    # Destroyed
#     }

#     rgba_mask = np.zeros((original_size[1], original_size[0], 4), dtype=np.uint8)
#     for class_idx, color in color_map.items():
#         rgba_mask[preds_resized == class_idx] = color

#     # 7. Overlay the mask onto the original optical image
#     opt_img_rgba = opt_img_raw.convert("RGBA")
#     mask_img = Image.fromarray(rgba_mask)
#     overlay_img = Image.alpha_composite(opt_img_rgba, mask_img)

#     return overlay_img.convert("RGB") # Streamlit prefers standard RGB


# --- 3. STREAMLIT GUI DESIGN ---
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

if opt_file is not None and sar_file is not None:
    if st.button("🚀 GENERATE COLOUR-GRADED MAP", use_container_width=True):
        
        with st.spinner("Running Siamese U-Net Inference..."):
            result_map = run_unet_inference(opt_file, sar_file)
            
            st.write("---")
            
            # Step 3: Output
            st.subheader("Step 3: Assessment Results")
            
            # Display the resulting map
            st.image(result_map, caption="Damage Map Overlay", use_container_width=True)
            
            # Display the Updated 5-Class Legend
            st.markdown("""
            **Map Legend:** 🟩 **Intact** &nbsp;&nbsp; 🟨 **Minor Damage** &nbsp;&nbsp; 🟧 **Major Damage** &nbsp;&nbsp; 🟥 **Destroyed**
            """)
            
            # Convert the PIL image to bytes for download
            buf = io.BytesIO()
            result_map.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="📥 Download Map (.png)",
                data=byte_im,
                file_name="unet_damage_map.png",
                mime="image/png"
            )
else:
    st.info("Please upload both Pre-Event and Post-Event images to unlock the assessment button.")