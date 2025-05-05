from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io
import os
import tensorflow as tf

from autoencoder import Autoencoder 

autoencoder = tf.keras.models.load_model('autoencoder_model.keras', custom_objects={'Autoencoder': Autoencoder})

app = FastAPI()

image_directory = 'images'

class AnomalyResponse(BaseModel):
    mse: float
    status: str
    reconstructed_image: bytes

def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((28, 28))
    image_array = np.array(image).astype('float32') / 255
    image_array = np.reshape(image_array, (1, 28, 28, 1))
    return image_array, image

@app.get("/predict_anomaly/{image_name}", response_model=AnomalyResponse)
async def predict_anomaly(image_name: str):
    image_path = os.path.join(image_directory, image_name)
    
    if not os.path.exists(image_path):
        return {"error": "Image not found"}
    
    image_array, original_image = preprocess_image(image_path)

    reconstructed = autoencoder.predict(image_array)

    mse = np.mean(np.square(image_array - reconstructed), axis=(1, 2, 3))

    threshold = 0.02
    status = "Anomaly" if mse > threshold else "Normal"

    reconstructed_image_pil = Image.fromarray((reconstructed[0] * 255).astype(np.uint8))
    img_byte_arr = io.BytesIO()
    reconstructed_image_pil.save(img_byte_arr, format='PNG')
    reconstructed_image_bytes = img_byte_arr.getvalue()

    return AnomalyResponse(mse=mse[0], status=status, reconstructed_image=reconstructed_image_bytes)
