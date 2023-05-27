"""
Custom Layers
"""

import tensorflow as tf

from .utils import squash

class PrimaryCapsules(tf.keras.layers.Layer):
    """
    The first layer of Capsules (Sabour et al, 2017)
    """
    def __init__(self, dim_capsule: int, n_capsule: int, kernel_size: int, strides: int):
        """
        `dim_capsule`: dimensionality of each capsule
        `n_capsule`: number of capsule channels
        `kernel_size`: kernel size of capsule convolution
        `strides`: stride of capsule convolution
        """
        super().__init__()

        self.dim_capsule = dim_capsule
        self.n_capsule = n_capsule
        self.kernel_size = kernel_size
        self.strides = strides

        # Layers
        self.conv = tf.keras.layers.Conv2D(
            filters=dim_capsule*n_capsule,
            kernel_size=kernel_size,
            strides=strides,
        )
        self.reshape = tf.keras.layers.Reshape(
            [n_capsule, -1]
        )

    def call(self, inputs):
        """
        `inputs`: Tensor of shape [batch, width, height, channels]
        Returns: Tensor of shape [batch, n_capsule, new_height * new_width * dim_capsule]
        """
        out = self.conv(inputs)
        out = self.reshape(out)
        return squash(out)


class CapsuleLayer(tf.keras.layers.Layer):
    """
    Capsule Layer with Dynamic Routing (Sabour et al, 2017)
    """
    def __init__(self, n_capsule: int, dim_capsule: int, n_routings: int):
        """
        `n_capsule`: Number of output capsules
        `dim_capsule`: Dimensionality of output capsules
        `n_routing`: Number of routing iterations (dynamic routing)
        """
        super().__init__()

        self.n_capsule = n_capsule
        self.dim_capsule = dim_capsule
        self.n_routings = n_routings

    def build(self, input_shape):
        """
        Build the layer weight and bias.
        """
        # input_shape[0] is the batch dimension
        self.n_input_capsule = input_shape[1]
        self.dim_input_capsule = input_shape[2]

        # weight and bias matrices
        # similar to normal FC layer where the input dim would be
        # equal to n_input_capsule * dim_input_capsule; and similarly
        # for the output dim.
        self.W = self.add_weight(
            shape=[
                self.n_input_capsule,
                self.n_capsule,
                self.dim_input_capsule,
                self.dim_capsule
            ],
            initializer='glorot_uniform',
            trainable=True,
            name="capsule_layer_weight"
        )
        self.B = self.add_weight(
            shape=[self.n_capsule, self.dim_capsule],
            initializer="zeros",
            trainable=True,
            name="capsule_layer_bias"
        )

    def call(self, inputs):
        """
        `inputs`: Tensor of shape [batch, n_input_capsule, dim_input_capsule]
        Returns: Tensor of shape [batch, n_capsule, dim_capsule]
        """
        batch_size = tf.shape(inputs)[0]

        # matmul like a normal FC layer. 
        #   tiled inputs has shape: [batch, n_input_capsule, n_capsule, 1, dim_input_capsule]
        #   tiled weights has shape: [batch, n_input_capsule, n_capsule, dim_input_capsule, dim_capsule]
        #   result has shape: [batch, n_input_capsule, n_capsule, 1, dim_capsule]
        u = tf.matmul(
            tf.tile(inputs[:, :, tf.newaxis, tf.newaxis, :], [1, 1, self.n_capsule, 1, 1]), 
            tf.tile(self.W[tf.newaxis, :, :, :, :], [batch_size, 1, 1, 1, 1])
        )
        u = tf.squeeze(u, axis=-2) # [batch, n_input_capsule, n_capsule, dim_capsule]
        u = u + self.B[tf.newaxis, tf.newaxis, :, :]

        # Dynamic Routing - coupling coefficients
        b = tf.zeros(shape=[batch_size, self.n_input_capsule, self.n_capsule], dtype=u.dtype)

        for _ in range(self.n_routings):
            c = tf.nn.softmax(b, axis=-1)
            s = tf.reduce_sum(u * c[:, :, :, tf.newaxis], axis=1) # [batch, n_capsule, dim_capsule]
            v = squash(s)

            b = b + tf.reduce_sum(u * v[:, tf.newaxis, :, :], axis=-1)
        
        return v
