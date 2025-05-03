import tensorflow as tf
from tensorflow.keras import layers, models

class Autoencoder(tf.keras.Model):
    def __init__(self, latent_dim, **kwargs):  # Pass other arguments to parent class
        super(Autoencoder, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
                layers.Flatten(input_shape=(28, 28)),
                layers.Dense(32, activation="relu"),
                layers.Dense(16, activation="relu"),
                layers.Dense(8, activation="relu"),
                layers.Dense(latent_dim, activation="relu")
            ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(8, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(784, activation="sigmoid"),
            layers.Reshape((28, 28))
            ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_config(self):
        # Return the configuration of the model, including the latent_dim
        config = super(Autoencoder, self).get_config()  # Get the parent class config
        config.update({'latent_dim': self.latent_dim})  # Add custom parameters (latent_dim)
        return config

    @classmethod
    def from_config(cls, config):
        # Recreate the model from its configuration
        # Make sure latent_dim is included in the configuration dictionary
        latent_dim = config.get('latent_dim', 64)  # Use default 64 if not found
        return cls(latent_dim=latent_dim)  # Pass the latent_dim to the constructor
