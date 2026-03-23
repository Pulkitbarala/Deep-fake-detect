import streamlit as st
import numpy as np
import cv2
from utils.predict import predict

st.set_page_config(page_title="Deepfake Detector", layout="centered")

st.title("Deepfake Image Detection System")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, channels="BGR", caption="Uploaded Image")

    if st.button("Analyze"):
        label, confidence = predict(img)

        st.subheader("Result:")
        st.write("Prediction:", label)
        st.write("Confidence:", round(confidence * 100, 2), "%")