import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from PIL import Image
import os
from autoencoder import Autoencoder

autoencoder = tf.keras.models.load_model('autoencoder_model.keras', custom_objects={'Autoencoder': Autoencoder})

app = FastAPI()

image_directory = 'images'

image_files = [f for f in os.listdir(image_directory) if f.endswith(('.png', '.jpg', '.jpeg'))]

def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((28, 28))
    image_array = np.array(image).astype('float32') / 255
    image_array = np.reshape(image_array, (1, 28, 28, 1))
    return image_array

class AnomalyResponse(BaseModel):
    mse: float
    status: str

@app.get("/predict_anomaly/{image_name}", response_model=AnomalyResponse)
async def predict_anomaly(image_name: str):
    if image_name not in image_files:
        return {"error": "Image not found"}

    image_path = os.path.join(image_directory, image_name)
    image_array = preprocess_image(image_path)

    reconstructed = autoencoder.predict(image_array)

    mse = np.mean(np.square(image_array - reconstructed), axis=(1, 2, 3))


    with open('threshold_value.txt', 'r') as f:
        threshold = float(f.read())
    status = "Anomaly" if mse > threshold else "Normal"

    return AnomalyResponse(mse=mse[0], status=status)
