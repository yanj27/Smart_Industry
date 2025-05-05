import streamlit as st
import requests
import os
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from autoencoder import Autoencoder
import cv2
import matplotlib.pyplot as plt

autoencoder = tf.keras.models.load_model('autoencoder_model.keras', custom_objects={'Autoencoder': Autoencoder})

image_directory = 'images'

image_files = [f for f in os.listdir(image_directory) if f.endswith(('.png', '.jpg', '.jpeg'))]

st.title('Autoencoder Anomaly Detection via API')
st.write("Select an image to check if it's an anomaly based on reconstruction error.")

selected_image = st.selectbox("Select Image", image_files)

if selected_image:
    image_path = f"{image_directory}/{selected_image}"

    original_image = Image.open(image_path)
    st.image(original_image, caption=f"Original Image: {selected_image}", use_container_width=True)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, (28, 28))
    image_resized = image_resized.astype('float32') / 255
    image_resized = np.reshape(image_resized, (1, 28, 28, 1))

    encoded_image = autoencoder.encoder(image_resized).numpy()
    decoded_image = autoencoder.decoder(encoded_image).numpy()

    decoded_image = np.squeeze(decoded_image)
    decoded_image = np.clip(decoded_image * 255, 0, 255).astype(np.uint8)

    st.write("Encoder")
    plt.imshow(image_resized[0, :, :, 0], cmap='gray')
    st.pyplot(plt)
    st.write("Decoder")
    plt.imshow(decoded_image, cmap='gray')
    st.pyplot(plt)

