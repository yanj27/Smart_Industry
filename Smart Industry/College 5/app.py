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
API_URL = "http://127.0.0.1:8000/predict_anomaly/"
image_directory = 'images'
images = []
image_files = [f for f in os.listdir(image_directory) if f.endswith(('.png', '.jpg', '.jpeg'))]

def anonymize_face_blur(image, blur_size=(50, 50)):
    # Get the height and width of the image
    h, w = image.shape[:2]

    # Define the center of the image
    centerX, centerY = w // 2, h // 2

    # Define the size of the region to blur (can adjust the size of the center blur)
    startX = centerX - blur_size[0] // 2
    startY = centerY - blur_size[1] // 2
    endX = centerX + blur_size[0] // 2
    endY = centerY + blur_size[1] // 2

    # Extract the region of interest (ROI) centered in the image
    roi = image[startY:endY, startX:endX]

    # Apply Gaussian blur to the ROI
    blurred_roi = cv2.GaussianBlur(roi, (15, 15), 0)

    # Replace the original region with the blurred ROI
    image[startY:endY, startX:endX] = blurred_roi

    return image

def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((28, 28))
    image_array = np.array(image).astype('float32') / 255
    image_array = np.reshape(image_array, (1, 28, 28, 1))
    return image_array      


st.title('Autoencoder Anomaly Detection via API')
st.write("Select an image to check if it's an anomaly based on reconstruction error.")

selected_image = st.selectbox("Select Image", image_files)

if selected_image:
    image_path = f"{image_directory}/{selected_image}"

    original_image = Image.open(image_path)
    st.image(original_image, caption=f"Original Image: {selected_image}", use_container_width=True)

    image = preprocess_image(image_path)

    image_resized = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image_resized, (28, 28))
    image_resized = image_resized.astype('float32') / 255
    image_resized = np.reshape(image_resized, (1, 28, 28, 1))

    encoded_image = autoencoder.encoder(image_resized).numpy()
    decoded_image = autoencoder.decoder(encoded_image).numpy()

    decoded_image = np.squeeze(decoded_image)

    pred = autoencoder.predict(image)
    pred = np.squeeze(pred)

    response = requests.get(f"{API_URL}{selected_image}")

    if response.status_code == 200:
        result = response.json()
        mse = result['mse']
        status = result['status']
        
        st.write(f"Reconstruction Error (MSE): {mse:.4f}")
        st.write(f"Status: {status}")
        if status == 'Normal':
                st.header("Normal")
                plt.imshow(pred, cmap='gray')
                st.pyplot(plt)
        else:
                st.header("Anomaly")
                img = cv2.imread(image_path, cv2.IMREAD_COLOR)
                
                img_blurred = anonymize_face_blur(img, blur_size=(100, 100))
                
                img_rgb = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2RGB)
                
                plt.imshow(img_rgb)
                st.pyplot(plt)
    else:
        st.error("Error occurred while processing the image.")
