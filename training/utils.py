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


def get_true_and_pred(capsule_reps, labels):
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


def linear_weight_to_correct_preds(y_true, y_pred):
    """
    `y_true`: Tensor of shape = [batch, n_classes]
        Binary labels [0, 1]
    `y_pred`: Tensor of shape = [batch, n_classes]
        Logits for each class.

    Returns: Tensor of shape [batch]
        Where each value is:
            0 where that batch sample was not correctly predicted
            1/n_correct otherwise.
        n_correct is the total number of correctly predicted batch samples.
    """
    correct = tf.cast(
        tf.argmax(y_true, axis=-1) == tf.argmax(y_pred, axis=-1),
        tf.float32
    )
    n_correct = tf.reduce_sum(correct)

    return correct/n_correct


def apply_training_mask(capsule_reps, labels):
    """
    Apply a training mask on the capsule representations
    for training the generators.

    `capsule_reps`: Output from classifier.
        Tensor of shape [batch, n_classes, dim_output]
    `labels: Integer labels. 
        Tensor of shape [batch]
    Returns: Tensor of shape [batch, n_classes, dim_output]
        Which is the same as `capsule_reps`, all zeros except
        for the slice along the dimension 1 corresponding to
        the correct label.
    """
    n_classes = tf.shape(capsule_reps)[1]

    labels_one_hot = tf.one_hot(labels, depth=n_classes, dtype=capsule_reps.dtype)

    return capsule_reps * labels_one_hot[:, :, tf.newaxis]
