"""
Custom functions
"""

import tensorflow as tf

def squash(inputs, axis=-1):
    """
    Squashing function (Sabour et al, 2017)
    `inputs`: Tensor of any shape
    Returns: Tensor of same shape as inputs
    """
    squared_norm = tf.math.reduce_sum(tf.math.square(inputs), axis=axis, keepdims=True)
    return squared_norm / (1 + squared_norm) * inputs / tf.math.sqrt(squared_norm)
