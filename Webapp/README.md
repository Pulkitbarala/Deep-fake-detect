# Deepfake Image Detection System

A Streamlit app for detecting deepfake images using a TensorFlow model. Upload an image, click Analyze, and get a Real/Fake prediction with confidence.

## Table of Contents
- Overview
- Features
- Project Structure
- Setup
- Run the App
- Usage
- How It Works
- Configuration Notes
- Model Management
- Performance Tips
- Troubleshooting
- FAQ

## Overview
This project provides a minimal, end-to-end inference pipeline for image-based deepfake detection. The UI is built with Streamlit and the model is loaded with TensorFlow/Keras.

## Features
- Upload JPG/PNG images and run inference
- Shows prediction label (Real/Fake) with confidence
- Simple, single-page Streamlit UI
- Modular preprocessing and prediction utilities

## Project Structure
```
.
├─ app.py
├─ requirements.txt
├─ model/
│  ├─ model.h5
│  └─ deepfake_efficientb4_attn_final.keras
└─ utils/
   ├─ preprocess.py
   └─ predict.py
```

## Setup
1. Create and activate a Python environment (recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Optional: Create a Virtual Environment
Windows PowerShell:
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## Run the App
```bash
streamlit run app.py
```

## Usage
- Open the Streamlit URL printed in the terminal.
- Upload a JPG/PNG image.
- Click **Analyze** to get the prediction and confidence.

## How It Works
1. **Upload**: Streamlit reads the uploaded file as bytes.
2. **Decode**: OpenCV converts bytes into a BGR image array.
3. **Preprocess**: Image is resized to 224x224, normalized to [0, 1], and batched.
4. **Inference**: The Keras model outputs a single probability value.
5. **Postprocess**: A threshold at 0.5 maps the probability to Fake/Real and computes confidence.

### Confidence Logic
- If prediction > 0.5: label = Fake, confidence = prediction
- Else: label = Real, confidence = 1 - prediction

## Configuration Notes
- The model is loaded once at import time in `utils/predict.py` for performance.
- `tf.keras.backend.clear_session()` is called to avoid stale graph state.
- Image preprocessing lives in `utils/preprocess.py` for easy reuse.

## Model Management
### Default Model
The app loads `model/model.h5` at startup.

### Replace the Model
1. Export or convert a compatible Keras model to `.h5`.
2. Replace `model/model.h5` with your new file.
3. Ensure your model accepts input shape 224x224x3 and outputs a single probability.

### About the .keras File
The `model/deepfake_efficientb4_attn_final.keras` file is present but not used by default. If you want to use it, update the load path in `utils/predict.py` accordingly.

## Performance Tips
- Use reasonably sized images; all inputs are resized to 224x224.
- Keep the model file on local disk (not a slow network drive).
- If startup is slow, the model load time is the main factor.

## Troubleshooting
- **Model file not found**: Ensure `model/model.h5` exists and the path is correct.
- **Incompatible model**: The app expects a single output probability. Re-export your model accordingly.
- **OpenCV errors**: Confirm `opencv-python` is installed and the image file is valid.
- **Streamlit won't start**: Verify the environment has `streamlit` installed.

## FAQ
**Does this work on videos?**
No. This app is for single image inference only.

**Can I change the decision threshold?**
Yes. Update the `0.5` threshold in `utils/predict.py`.

**Is training included?**
No. This repository focuses on inference and a simple UI.
