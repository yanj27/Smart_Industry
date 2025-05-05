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
    def __init__(self, latent_dim, **kwargs):  
        super(Autoencoder, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        
        # Encoder
        self.flatten = layers.Flatten(input_shape=(28, 28))
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(latent_dim, activation='relu')  # Latent dimension
        
        # Decoder
        self.dense3 = layers.Dense(128, activation='relu')
        self.dense4 = layers.Dense(784, activation='sigmoid')
        self.reshape = layers.Reshape((28, 28))  # Reconstruct back to original shape

    def call(self, x):
        encoded = self.flatten(x)
        encoded = self.dense1(encoded)
        encoded = self.dense2(encoded)
        
        decoded = self.dense3(encoded)
        decoded = self.dense4(decoded)
        decoded = self.reshape(decoded)
        
        return decoded

    def get_config(self):
        config = super(Autoencoder, self).get_config()
        config.update({'latent_dim': self.latent_dim})
        return config

    @classmethod
    def from_config(cls, config):
        latent_dim = config.get('latent_dim', 64)
        return cls(latent_dim=latent_dim)

# Train the Autoencoder model
def train_autoencoder(x_train, x_test, latent_dim, num_epochs):
    autoencoder = Autoencoder(latent_dim)

    autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

    history = autoencoder.fit(x_train, x_train, 
                              epochs=num_epochs,
                              shuffle=True,
                              validation_data=(x_test, x_test))

    # Save the model after training
    autoencoder.save('autoencoder_model.keras')
    return autoencoder

# Load MNIST dataset
x_train, x_test = train_test_split(images, test_size=0.2, random_state=42)

# Normalize the images (0-1)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Train and save the model
train_autoencoder(x_train, x_test, latent_dim=64, num_epochs=10)
