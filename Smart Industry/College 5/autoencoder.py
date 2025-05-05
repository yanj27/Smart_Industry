import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers, models


def load_data(image_folder, target_size=(28, 28)):
    images = []
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        
        if image_name.endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(image_path).convert('L')
            image = image.resize(target_size)
            
            image_array = np.array(image).astype('float32') / 255
            images.append(image_array)
    
    return np.array(images)

image_folder = 'image_data'
images = load_data(image_folder, target_size=(28, 28))
data = images.reshape(-1, 28, 28, 1)

@tf.keras.utils.register_keras_serializable()
class Autoencoder(tf.keras.Model):
    def __init__(self, latent_dim=64):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(28, 28, 1)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(latent_dim, activation='relu')
        ])

        # Decoder
        self.decoder = tf.keras.Sequential([
            layers.Dense(64, activation='relu', input_dim=latent_dim),
            layers.Dense(128, activation='relu'),
            layers.Dense(784, activation='sigmoid'),
            layers.Reshape((28, 28, 1))
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
        latent_dim = config.get('latent_dim', 64)
        return cls(latent_dim=latent_dim)


def train_autoencoder(x_train, x_test, latent_dim, num_epochs):
    autoencoder = Autoencoder(latent_dim)

    autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

    history = autoencoder.fit(x_train, x_train, 
                              epochs=num_epochs,
                              shuffle=True,
                              validation_data=(x_test, x_test))

    autoencoder.save('autoencoder_model.keras')
    return autoencoder

x_train, x_test = train_test_split(images, test_size=0.2, random_state=42)

x_train = x_train / 255
x_test = x_test / 255 

train_autoencoder(x_train, x_test, latent_dim=64, num_epochs=150)
