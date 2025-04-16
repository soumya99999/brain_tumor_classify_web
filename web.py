import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown
import os

# Google Drive file ID for the model (replace with your actual file ID)
MODEL_FILE_ID = '1NUWfYhGall3DnCcemnWLgw0n5GXqGbus'
MODEL_PATH = '1024_relu_xception_model.keras'

# Download the model from Google Drive if not already present
if not os.path.exists(MODEL_PATH):
    url = f'https://drive.google.com/uc?id={MODEL_FILE_ID}'
    gdown.download(url, MODEL_PATH, quiet=False)

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Define class names
class_names = ['Glioma', 'No tumor']

st.title("Brain Tumor Classification")
st.write("Upload an MRI image to classify as Glioma or No tumor.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)  # Updated parameter

    # Preprocess the image to match Colab (299x299, normalized)
    img = image.resize((299, 299), Image.Resampling.LANCZOS)  # Explicitly use LANCZOS for resizing
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1], ensure float32
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Debugging: Display raw prediction output
    predictions = model.predict(img_array)
    st.write(f"Raw prediction probability: {predictions[0][0]:.4f}")

    # Interpret prediction (sigmoid output, threshold at 0.5)
    predicted_class = class_names[1] if predictions[0][0] > 0.5 else class_names[0]
    st.write(f"Prediction: *{predicted_class}*")