import streamlit as st
import requests
import os
import numpy as np
from PIL import Image
import io

API_URL = "http://127.0.0.1:8000/predict_anomaly/"

image_directory = 'images'

image_files = [f for f in os.listdir(image_directory) if f.endswith(('.png', '.jpg', '.jpeg'))]

st.title('Autoencoder Anomaly Detection via API')
st.write("Select an image to check if it's an anomaly based on reconstruction error.")


selected_image = st.selectbox("Select Image", image_files)

if selected_image:
    image_path = f"{image_directory}/{selected_image}"
    original_image = Image.open(image_path)
    st.image(original_image, caption=f"Original Image: {selected_image}", use_container_width=True)

    response = requests.get(f"{API_URL}{selected_image}")

    if response.status_code == 200:
        result = response.json()
        mse = result['mse']
        status = result['status']

        reconstructed_image_data = result['reconstructed_image']
        reconstructed_image = Image.open(io.BytesIO(reconstructed_image_data))

        st.image(reconstructed_image, caption=f"Reconstructed Image: {selected_image}", use_column_width=True)

        st.write(f"Reconstruction Error (MSE): {mse:.4f}")
        st.write(f"Status: {status}")
    else:
        st.error("Error occurred while processing the image.")
