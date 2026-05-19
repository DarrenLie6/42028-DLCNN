# import streamlit as st
# import numpy as np
# from PIL import Image
# import io
# import sys
# import os
# import torch
# import torchvision.transforms as transforms

# # Force Python to look in the current folder for modules (if needed for imports)
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from src.models.siamese_unet import SiameseUNet

# # --- 1. MODEL LOADING ---
# @st.cache_resource
# def load_model():
#     # Initialize the architecture with 5 classes to match your EDA findings
#     # (0: Background, 1: Intact, 2: Minor, 3: Major, 4: Destroyed)
#     model = SiameseUNet(num_classes=5)
    
#     # Load trained weights if available
#     weight_path = "checkpoints/UNet.pth"
#     try:
#         model.load_state_dict(torch.load(weight_path, map_location="cpu"))
#         print("Loaded trained model weights.")
#     except FileNotFoundError:
#         print("No trained weights found. Using random initialization.")
    
#     model.eval() # Set to evaluation mode
#     return model

# # --- 2. INFERENCE SCRIPT ---
# def run_unet_inference(post_img_file, pre_img_file=None):
#     model = load_model()

#     # 1. Load Post-Event image (Required for model)
#     post_img_raw = Image.open(post_img_file).convert("RGB")
#     original_size = post_img_raw.size # (Width, Height)

#     # 2. Preprocessing (Resize to 256x256 and convert to tensor)
#     # Using standard ImageNet normalization for optical
#     opt_transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     # 3. Apply transforms and add batch dimension (B, C, H, W)
#     post_tensor = opt_transform(post_img_raw).unsqueeze(0)

#     # 4. Run Model Inference (Only Post-Event is passed through the network now)
#     with torch.no_grad():
#         logits = model(post_tensor)
        
#     preds = torch.argmax(logits, dim=1).squeeze(0).numpy() # Shape: (256, 256)

#     # Convert prediction array back to an image and resize it to original upload resolution
#     preds_img = Image.fromarray(preds.astype(np.uint8)).resize(original_size, Image.NEAREST)
#     preds_resized = np.array(preds_img)

#     # 5. Apply RGBA Color Map to predictions 
#     color_map = {
#         0: [0, 0, 0, 0],         # Background (Transparent)
#         1: [46, 204, 113, 150],  # Intact
#         2: [241, 196, 15, 150],  # Minor
#         3: [230, 126, 34, 150],  # Major
#         4: [231, 76, 60, 150]    # Destroyed
#     }

#     rgba_mask = np.zeros((original_size[1], original_size[0], 4), dtype=np.uint8)
#     for class_idx, color in color_map.items():
#         rgba_mask[preds_resized == class_idx] = color

#     mask_img = Image.fromarray(rgba_mask)

#     # 6. Determine Background for Overlay
#     if pre_img_file is not None:
#         # User provided a pre-event image, so overlay the damage map onto this
#         bg_img_raw = Image.open(pre_img_file).convert("RGBA")
#         # Safety check: force pre-image to match post-image dimensions if they differ
#         if bg_img_raw.size != original_size:
#             bg_img_raw = bg_img_raw.resize(original_size)
#     else:
#         # No pre-event image provided, overlay directly onto the post-event image
#         bg_img_raw = post_img_raw.convert("RGBA")

#     # 7. Apply Overlay
#     overlay_img = Image.alpha_composite(bg_img_raw, mask_img)

#     return overlay_img.convert("RGB") # Streamlit prefers standard RGB


# # --- 3. STREAMLIT GUI DESIGN ---
# st.set_page_config(page_title="ImpactVision", layout="centered")

# st.title("ImpactVision: Satellite Damage Assessment Tool")
# st.write("---")

# # Step 1: Uploads
# st.subheader("Step 1: Upload Satellite Imagery")
# col1, col2 = st.columns(2)

# with col1:
#     st.write("**Post-Event Image (Optical RGB) - REQUIRED**")
#     post_file = st.file_uploader("Drag & Drop Post-Event", type=["png", "jpg", "jpeg", "tif"])

# with col2:
#     st.write("**Pre-Event Image (Optical RGB) - OPTIONAL**")
#     pre_file = st.file_uploader("Drag & Drop Pre-Event", type=["png", "jpg", "jpeg", "tif"])

# st.write("---")

# # Step 2: Assessment Trigger
# st.subheader("Step 2: Run Assessment")

# # Now we only require the post_file to unlock the trigger button
# if post_file is not None:
#     if st.button("🚀 GENERATE COLOUR-GRADED MAP", use_container_width=True):
        
#         with st.spinner("Running U-Net Inference..."):
#             # Pass both the required post_file and the optional pre_file
#             result_map = run_unet_inference(post_img_file=post_file, pre_img_file=pre_file)
            
#             st.write("---")
            
#             # Step 3: Output
#             st.subheader("Step 3: Assessment Results")
            
#             # Adjust caption based on what background was used
#             caption_text = "Damage Map Overlay (on Pre-Event)" if pre_file else "Damage Map Overlay (on Post-Event)"
#             st.image(result_map, caption=caption_text, use_container_width=True)
            
#             # Display the 5-Class Legend
#             st.markdown("""
#             **Map Legend:** 🟩 **Intact** &nbsp;&nbsp; 🟨 **Minor Damage** &nbsp;&nbsp; 🟧 **Major Damage** &nbsp;&nbsp; 🟥 **Destroyed**
#             """)
            
#             # Convert the PIL image to bytes for download
#             buf = io.BytesIO()
#             result_map.save(buf, format="PNG")
#             byte_im = buf.getvalue()
            
#             st.download_button(
#                 label="📥 Download Map (.png)",
#                 data=byte_im,
#                 file_name="unet_damage_map.png",
#                 mime="image/png"
#             )
# else:
#     st.info("Please upload a Post-Event optical image to unlock the assessment button.")

# == FastAPI Version ==
import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="ImpactVision", layout="centered")

st.title("ImpactVision: Satellite Damage Assessment Tool")
st.write("---")

# Step 1: Uploads
st.subheader("Step 1: Upload Satellite Imagery")
col1, col2 = st.columns(2)

with col1:
    st.write("**Post-Event Image (Optical RGB) - REQUIRED**")
    post_file = st.file_uploader("Drag & Drop Post-Event", type=["png", "jpg", "jpeg", "tif"])

with col2:
    st.write("**Pre-Event Image (Optical RGB) - OPTIONAL**")
    pre_file = st.file_uploader("Drag & Drop Pre-Event", type=["png", "jpg", "jpeg", "tif"])

st.write("---")

# Step 2: Assessment Trigger
st.subheader("Step 2: Run Assessment")

# Only require the post_file to unlock the trigger button
if post_file is not None:
    if st.button("🚀 GENERATE COLOUR-GRADED MAP", use_container_width=True):
        
        with st.spinner("Sending image(s) to FastAPI Server..."):
            
            # Always send the required post-event file
            files = {
                "post_file": (post_file.name, post_file.getvalue(), post_file.type)
            }
            
            # If the user included the optional pre-event file, add it to the payload
            if pre_file is not None:
                files["pre_file"] = (pre_file.name, pre_file.getvalue(), pre_file.type)
            
            try:
                # Make the POST request to our local FastAPI server
                response = requests.post("http://127.0.0.1:8000/predict", files=files)
                response.raise_for_status() 
                
                # Convert the returned bytes back into an image
                result_map = Image.open(io.BytesIO(response.content))
                
                st.write("---")
                
                # Step 3: Output
                st.subheader("Step 3: Assessment Results")
                
                # Adjust caption based on what background was used
                caption_text = "Damage Map Overlay (on Pre-Event)" if pre_file else "Damage Map Overlay (on Post-Event)"
                st.image(result_map, caption=caption_text, use_container_width=True)
                
                # Display the 5-Class Legend
                st.markdown("""
                **Map Legend:** 🟩 **Intact** &nbsp;&nbsp; 🟨 **Minor Damage** &nbsp;&nbsp; 🟧 **Major Damage** &nbsp;&nbsp; 🟥 **Destroyed**
                """)
                
                # Download Button
                st.download_button(
                    label="📥 Download Map (.png)",
                    data=response.content,
                    file_name="unet_damage_map.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"Failed to connect to the FastAPI server. Ensure `uvicorn api:app` is running. Error: {e}")
else:
    st.info("Please upload a Post-Event optical image to unlock the assessment button.")