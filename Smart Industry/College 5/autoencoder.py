import numpy as np
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
            images.append(image)
    random.shuffle(images)
    return np.stack(images)

image_folder = 'image_data'
images = load_data(image_folder)

@tf.keras.utils.register_keras_serializable()
class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(784, activation='sigmoid'),
            layers.Reshape((28, 28))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_config(self):
        config = super(Autoencoder, self).get_config()
        config.update({'latent_dim': self.latent_dim})
        return config

    @classmethod
    def from_config(cls, config):
        latent_dim = config.get('latent_dim', 164)
        return cls(latent_dim=latent_dim)

def train_autoencoder(x_train, x_test, latent_dim, num_epochs):
    autoencoder = Autoencoder(latent_dim)

    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

    history = autoencoder.fit(x_train, x_train, 
                              epochs=num_epochs,
                              shuffle=True,
                              validation_data=(x_test, x_test))
    
    return autoencoder, history.history['loss'], history.history['val_loss']

x_train, x_test = train_test_split(images, test_size=0.2, random_state=42)

x_train = x_train / 255
x_test = x_test / 255 

model, loss_train, loss_test = train_autoencoder(x_train, x_test, 164, 150)

reconstructed = model.predict(images)
mse_scores = [mean_squared_error(images[i].flatten() / 255, reconstructed[i].flatten() / 255)
              for i in range(images.shape[0])]

threshold = np.mean(mse_scores) + np.std(mse_scores)
with open('threshold_value.txt', 'w') as f:
    f.write(str(threshold))

model.save('autoencoder_model.keras')
