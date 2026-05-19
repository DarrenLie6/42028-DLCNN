# api.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from typing import Optional
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import sys
import os

# Force Python to look in the current folder for modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.models.siamese_unet import SiameseUNet

app = FastAPI(title="ImpactVision API")

def load_model():
    # Initialize the architecture with 5 classes
    model = SiameseUNet(num_classes=5)
    
    # Load trained weights if available
    weight_path = "checkpoints/UNet.pth"
    try:
        model.load_state_dict(torch.load(weight_path, map_location="cpu"))
        print("Loaded trained model weights.")
    except FileNotFoundError:
        print("No trained weights found. Using random initialization.")
    
    model.eval()
    return model

# Load model globally when API starts
model = load_model()

@app.post("/predict")
async def predict_damage(
    post_file: UploadFile = File(...),              # Required
    pre_file: Optional[UploadFile] = File(None)     # Optional
):
    # 1. Read Post-Event File
    post_bytes = await post_file.read()
    post_img_raw = Image.open(io.BytesIO(post_bytes)).convert("RGB")
    original_size = post_img_raw.size

    # 2. Preprocessing (Optical Post-Event Only)
    opt_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    post_tensor = opt_transform(post_img_raw).unsqueeze(0)

    # 3. Model Inference
    # with torch.no_grad():
    #     logits = model(post_tensor)
        
    # preds = torch.argmax(logits, dim=1).squeeze(0).numpy()
    # preds_img = Image.fromarray(preds.astype(np.uint8)).resize(original_size, Image.NEAREST)
    # preds_resized = np.array(preds_img)

    # 3. Model Inference (Passing dummy data to satisfy the old Siamese architecture)
    with torch.no_grad():
        dummy_sar = torch.zeros((1, 1, 256, 256))
        dummy_valid = torch.tensor([1.0])
        logits = model(post_tensor, dummy_sar, dummy_valid)
        
    # ---> THESE ARE THE MISSING LINES <---
    # Convert logits to class predictions (0, 1, 2, 3, 4)
    preds = torch.argmax(logits, dim=1).squeeze(0).numpy()
    
    # Resize the 256x256 prediction back to the original image upload size
    preds_img = Image.fromarray(preds.astype(np.uint8)).resize(original_size, Image.NEAREST)
    preds_resized = np.array(preds_img)
    # -------------------------------------

    # 4. Color Mapping
    color_map = {
        0: [0, 0, 0, 0],         
        1: [46, 204, 113, 150],  
        2: [241, 196, 15, 150],  
        3: [230, 126, 34, 150],  
        4: [231, 76, 60, 150]    
    }
    
    rgba_mask = np.zeros((original_size[1], original_size[0], 4), dtype=np.uint8)
    for class_idx, color in color_map.items():
        rgba_mask[preds_resized == class_idx] = color

    mask_img = Image.fromarray(rgba_mask)

    # 5. Determine Background for Overlay
    if pre_file is not None:
        # If user provided a pre-event image, overlay on that
        pre_bytes = await pre_file.read()
        bg_img_raw = Image.open(io.BytesIO(pre_bytes)).convert("RGBA")
        # Ensure pre-image matches the exact dimensions of post-image
        if bg_img_raw.size != original_size:
            bg_img_raw = bg_img_raw.resize(original_size)
    else:
        # If no pre-event image, overlay directly onto the post-event image
        bg_img_raw = post_img_raw.convert("RGBA")

    # 6. Apply Overlay and Return
    overlay_img = Image.alpha_composite(bg_img_raw, mask_img).convert("RGB")

    buf = io.BytesIO()
    overlay_img.save(buf, format="PNG")
    
    return Response(content=buf.getvalue(), media_type="image/png")