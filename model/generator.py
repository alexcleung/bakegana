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
        output_shape: Tuple[int, int, int]
    ):
        """
        `output_shape`: (width, height, channels)
        """
        super().__init__()

        self.output_shape = output_shape

        # Layers
        self.fc = tf.keras.layers.Dense(
            output_shape[0]*output_shape[1]*output_shape[2],
            activation="relu"
        )
        self.reshape = tf.keras.layers.Reshape(
            output_shape
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
            strides=2,
            padding="same",
            activation="relu"
        )
        self.deconv3 = tf.keras.layers.Conv2DTranspose(
            filters=8,
            kernel_size=3,
            strides=2,
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
        Returns: Tensor of shape `output_shape`
        """
        x = self.fc(inputs)
        x = self.reshape(x)
        x = self.bn(x, training=training)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)

        return x
