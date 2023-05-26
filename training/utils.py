"""
Loss functions
"""

import tensorflow as tf

def get_pred(capsule_reps):
    """
    `capsule_reps`: Output from classifier.
        Tensor of shape [batch, n_classes, dim_output]
    Returns: logits of predicted class
        Tensor of shape [batch, n_classes]
    """
    return tf.math.sqrt(
        tf.reduce_sum(
            tf.math.square(capsule_reps),
            axis=-1
        )
    )

def get_pred_and_label(capsule_reps, labels):
    """
    Create y_pred and y_true to pass to tf.keras.losses.Loss

    `capsule_reps`: Output from classifier.
        Tensor of shape [batch, n_classes, dim_output]
    `labels: Integer labels. 
        Tensor of shape [batch]
    Returns: Tensors of same shape = [batch, n_classes]
        y_true: Binary labels (0, 1)
        y_pred: Logits of predicted classes
    """
    n_classes = tf.shape(capsule_reps)[1]

    y_pred = get_pred(capsule_reps)

    y_true = tf.one_hot(labels, depth=n_classes, dtype=y_pred.dtype)

    return y_true, y_pred
