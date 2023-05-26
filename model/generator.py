"""
Generator Network based on Jayasundra et al., 2019
"""
from typing import Tuple

import tensorflow as tf

class KanaGenerator(tf.keras.Model):
    """
    Generate kana based on its capsule representation.
    """
    def __init__(
        self,
        image_shape: Tuple[int, int, int]
    ):
        """
        `image_shape`: (width, height, channels)
        """
        super().__init__()

        self.image_shape = image_shape

        # Layers
        self.reshape_1 = tf.keras.layers.Reshape(
            [-1]
        )
        self.fc = tf.keras.layers.Dense(
            image_shape[0]*image_shape[1]*image_shape[2],
            activation="relu"
        )
        self.reshape_2 = tf.keras.layers.Reshape(
            image_shape
        )
        self.bn = tf.keras.layers.BatchNormalization()
        self.deconv1 = tf.keras.layers.Conv2DTranspose(
            filters=32,
            kernel_size=3,
            strides=1,
            padding="same",
            activation="relu"
        )
        self.deconv2 = tf.keras.layers.Conv2DTranspose(
            filters=16,
            kernel_size=3,
            strides=1,
            padding="same",
            activation="relu"
        )
        self.deconv3 = tf.keras.layers.Conv2DTranspose(
            filters=8,
            kernel_size=3,
            strides=1,
            padding="same",
            activation="relu"
        )
        self.deconv4 = tf.keras.layers.Conv2DTranspose(
            filters=4,
            kernel_size=3,
            strides=1,
            padding="same",
            activation="relu"
        )
        self.deconv5 = tf.keras.layers.Conv2DTranspose(
            filters=1,
            kernel_size=3,
            strides=1,
            padding="same",
            activation="sigmoid"
        )

    def call(self, inputs, training=False):
        """
        `inputs`: Tensor of shape [batch, n_classes, dim_output]
        Returns: Tensor of shape `image_shape`
        """
        x = self.reshape_1(inputs)
        x = self.fc(x)
        x = self.reshape_2(x)
        x = self.bn(x, training=training)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)

        return x


class KanaGenerator2(tf.keras.Model):
    """
    Simpler Decoder, with just a few FC layers.
    """
    def __init__(
        self,
        image_shape: Tuple[int, int, int]
    ):
        """
        `image_shape`: (width, height, channels)
        """
        super().__init__()

        self.image_shape = image_shape

        # Layers
        self.reshape_1 = tf.keras.layers.Reshape(
            [-1]
        )
        self.fc1 = tf.keras.layers.Dense(512, activation="relu")
        self.fc2 = tf.keras.layers.Dense(1024, activation="relu")
        self.fc3 = tf.keras.layers.Dense(
            image_shape[0]*image_shape[1]*image_shape[2],
            activation="sigmoid"
        )
        self.reshape_2 = tf.keras.layers.Reshape(image_shape)

    def call(self, inputs, training=False):
        """
        `inputs`: Tensor of shape [batch, n_classes, dim_output]
        Returns: Tensor of shape `image_shape`
        """
        x = self.reshape_1(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.reshape_2(x)

        return x
