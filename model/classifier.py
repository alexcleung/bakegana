"""
Classification Network based on CapsNet (Jayasundra et al., 2019)
"""

from typing import Tuple

import tensorflow as tf

from .layers import PrimaryCapsules, CapsuleLayer

class KanaClassifier(tf.keras.Model):
    """
    Classify an image input.
    """
    def __init__(
            self,
            n_classes: int,
            dim_output: int = 16,
            n_routings: int = 3
        ):
        """
        `n_classes`: Number of possible classes
        `dim_output`: Size of each class representation
        `n_routing`: Number of routing iterations (dynamic routing)
        """
        super().__init__()
        
        self.n_classes = n_classes
        self.n_routings = n_routings

        # Layers
        self.conv1 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            activation='relu',
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=3,
            activation='relu',
        )
        self.conv3 = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=3,
            strides=2,
            activation='relu',
        )
        self.primary_caps = PrimaryCapsules(
            dim_capsule=8,
            n_capsule=32,
            kernel_size=9,
            strides=2
        )
        self.kana_caps = CapsuleLayer(n_capsule=n_classes, dim_capsule=dim_output, n_routings=n_routings)

    def call(self, inputs, training=False):
        """
        `inputs`: Tensor of shape [batch, width, height, channels]
        Returns: 
            `kana_reps`: Tensor of shape [batch, n_classes, dim_output] representing the 
                capsule representation for each class.
            `scores`: Tensor of shape [batch, n_classes] representing the
                unnormalized probabilities of each class.
        """
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.primary_caps(x)
        kana_reps = self.kana_caps(x)
        
        return kana_reps
