import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
import cv2
import random


def load_data(image_folder):
    images = []
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        
        if image_name.endswith(('.png', '.jpg', '.jpeg')):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = Image.fromarray(image)
            image = np.array((image.resize((28,28), Image.LANCZOS)))
            # image = image.astype('float32') / 255.0
            # image = np.reshape(image, (1, 28, 28, 1))
            images.append(image)
    random.shuffle(images)
    return np.stack(images)

def add_gaussian_noise(image, mean=0, std=10):
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image


image_folder = 'image_data'
images = load_data(image_folder)

images = images.astype('float32') / 255.0
images = np.reshape(images, (-1, 28, 28, 1))

x_synth_noisy = np.array([add_gaussian_noise(img.squeeze()) for img in images])
x_synth_noisy = x_synth_noisy.astype('float32') / 255.0
x_synth_noisy = np.reshape(x_synth_noisy, (-1, 28, 28, 1))

# Flatten for feeding into the autoencoder
X_train = images
X_train_noisy = x_synth_noisy

def train_autoencoder(X_train_noisy, X_train):
    from tensorflow.keras import layers, models

    input_img = layers.Input(shape=(28, 28, 1))

    # Encoder (3 downsamples)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)  # -> 14x14
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)  # -> 7x7
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)  # -> 4x4

    # Decoder (3 upsamples)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)  # -> 8x8
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)  # -> 16x16
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)  # -> 32x32
    x = layers.Cropping2D(((2, 2), (2, 2)))(x)  # Crop back to 28x28
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = models.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    autoencoder.fit(
        X_train_noisy, X_train,
        epochs=50,
        batch_size=128,
        shuffle=True,
        validation_split=0.2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]
    )

    return autoencoder


autoencoder = train_autoencoder(X_train_noisy, X_train)


autoencoder.save('denoiser.keras')

