# ImpactVision - AI-Powered Satellite Damage Assessment

## Overview

ImpactVision is a deep learning application for automated disaster damage assessment using post-event satellite imagery. The system uses a custom Attention U-Net model trained for semantic segmentation of building damage and generates a colour-coded damage map overlay.

## Project Structure

project_root/
│
├── app/
│   └── streamlit_app.py
│
├── checkpoints/
│   └── xview2/
│       └── UNet.pth
│
├── configs/
│
├── src/
│   ├── data/
│   ├── evaluation/
│   └── models/
│
└── README.md

## Model Checkpoint

The trained model weights must be placed in:

checkpoints/xview2/UNet.pth

Create the folders if they do not already exist:

```bash
mkdir -p checkpoints/xview2
```

Then copy the trained checkpoint.

## Dependencies

Install the dependencies from the root directory using:

```bash
pip install -r requirements.txt
```

## Using the Application

From the project root directory:

```bash
streamlit run app/streamlit_app.py
```

After startup, Streamlit will display a local URL such as http://localhost:8501.

Open this URL in your browser.

1. Upload a post-disaster satellite image.
2. Click **GENERATE COLOUR-GRADED MAP**.
3. Wait for inference to complete.
4. View the generated damage assessment overlay.
5. Download the resulting PNG file if required.

Supported input formats include:

* PNG (`.png`)
* JPEG (`.jpg`, `.jpeg`)
* GeoTIFF (`.tif`, `.tiff`)
