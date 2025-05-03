import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from PIL import Image
import os
from tensorflow.keras.models import Model, load_model
from autoencoder import Autoencoder

# Load the model with custom_objects argument
autoencoder = load_model('autoencoder_model.h5', custom_objects={'Autoencoder': Autoencoder})

# Initialize FastAPI app
app = FastAPI()

# Path to the directory containing the images
image_directory = 'images'
image_files = [f for f in os.listdir(image_directory) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Preprocess image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image_array = np.array(image).astype('float32') / 255.0  # Normalize
    image_array = np.reshape(image_array, (1, 28, 28, 1))  # Add batch dimension
    return image_array

# Pydantic model for API response
class AnomalyResponse(BaseModel):
    mse: float
    status: str

# FastAPI endpoint for anomaly detection
@app.get("/predict_anomaly/{image_name}", response_model=AnomalyResponse)
async def predict_anomaly(image_name: str):
    if image_name not in image_files:
        return {"error": "Image not found"}

    image_path = os.path.join(image_directory, image_name)
    image_array = preprocess_image(image_path)

    # Predict reconstruction using the autoencoder
    reconstructed = autoencoder.predict(image_array)

    # Calculate reconstruction error (MSE)
    mse = np.mean(np.square(image_array - reconstructed), axis=(1, 2, 3))

    # Set threshold for anomaly detection
    threshold = 0.02
    status = "Anomaly" if mse > threshold else "Normal"

    return AnomalyResponse(mse=mse[0], status=status)
