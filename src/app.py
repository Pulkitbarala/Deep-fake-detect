import streamlit as st
import cv2
import numpy as np
from PIL import Image
from predict import Predictor
import tempfile
import os

st.set_page_config(page_title="Deepfake Detection", layout="centered")

st.title("Deepfake Detection")
st.write("Upload an image to detect if it's Real or Fake.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file to a temp file
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    # Display image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    if st.button("Analyze"):
        with st.spinner('Analyzing...'):
            predictor = Predictor("best_model.pth")
            
            # Read image for opencv
            image = Image.open(uploaded_file)
            image_np = np.array(image.convert('RGB')) 
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            label, confidence = predictor.predict(image_cv)
            
            color = "green" if label == "Real" else "red"
            st.markdown(f"<h2 style='text-align: center; color: {color};'>{label}</h2>", unsafe_allow_html=True)
            st.write(f"Confidence: {confidence:.2%}")

    # Cleanup
    tfile.close()
    os.unlink(tfile.name)
