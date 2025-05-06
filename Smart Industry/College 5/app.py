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
denoiser = tf.keras.models.load_model('denoiser.keras')
API_URL = "http://127.0.0.1:8000/predict_anomaly/"
image_directory = 'images'
images = []
image_files = [f for f in os.listdir(image_directory) if f.endswith(('.png', '.jpg', '.jpeg'))]

def add_gaussian_noise(image, mean=0, std=10):
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image


def pixelate_center(image, blur_percentage=0.2):
    # Get the height and width of the image
    h, w = image.shape[:2]

    # Define the center of the image
    centerX, centerY = w // 2, h // 2

    # Define the size of the region to pixelate (scaled based on image size)
    blur_width = int(w * blur_percentage)
    blur_height = int(h * blur_percentage)

    # Define the coordinates for the region to pixelate
    startX = centerX - blur_width // 2
    startY = centerY - blur_height // 2
    endX = centerX + blur_width // 2
    endY = centerY + blur_height // 2

    # Extract the region of interest (ROI) centered in the image
    roi = image[startY:endY, startX:endX]

    # Apply pixelation to the ROI by dividing the region into blocks
    block_size = 10  # Set block size (can adjust)
    for i in range(0, roi.shape[0], block_size):
        for j in range(0, roi.shape[1], block_size):
            # Get the block from the region
            block = roi[i:i+block_size, j:j+block_size]
            
            # Compute the average color of the block (mean pixel value)
            avg_color = np.mean(block, axis=(0, 1), dtype=int)
            
            # Set the block to the average color
            roi[i:i+block_size, j:j+block_size] = avg_color

    # Place the pixelated region back into the image
    image[startY:endY, startX:endX] = roi

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
            st.header("With Noise")
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28))
            noisy_image = add_gaussian_noise(img, mean=0, std=10)

            plt.figure()
            plt.imshow(noisy_image, cmap='gray')
            plt.axis('off')
            st.pyplot(plt)

            st.header("Without Noise")

            # Proper preprocessing for Conv2D model
            input_for_model = noisy_image.astype('float32') / 255.0
            input_for_model = input_for_model.reshape(1, 28, 28, 1)

            # Predict
            denoised = denoiser.predict(input_for_model)
            denoised = denoised.reshape(28, 28)

            plt.figure()
            plt.imshow(denoised, cmap='gray')
            plt.axis('off')
            st.pyplot(plt)

        else:
                st.header("Anomaly")
                img = cv2.imread(image_path, cv2.IMREAD_COLOR)
                
                img_blurred = pixelate_center(img, blur_percentage=0.5)
                
                img_rgb = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2RGB)
                
                plt.imshow(img_rgb)
                st.pyplot(plt)
    else:
        st.error("Error occurred while processing the image.")
