"""
Training the classifiers based on ground truth
"""
from typing import Dict

import tensorflow as tf

from ..model.classifier import KanaClassifier
from .utils import get_pred_and_label


def train(
    config: Dict,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    label_mapping: Dict 
):
    """
    Training script
    """
    # Declare models
    hiragana_classifier = KanaClassifier(
        input_shape=config["image_size"] + [1], # 1 channel (grayscale)
        n_classes=len(label_mapping),
    )
    katakana_classifier = KanaClassifier(
        input_shape=config["image_size"] + [1], # 1 channel (grayscale)
        n_classes=len(label_mapping),
    )

    # Optimizers for classifiers
    hiragana_optimizer = tf.keras.optimizers.get(
        config["optimizer_type"]
    )(**config["optimizer_config"])
    katakana_optimizer = tf.keras.optimizers.get(
        config["optimizer_type"]
    )(**config["optimizer_config"])

    #########################################
    #       LOSS FUNCTION AND METRICS       #
    #########################################
    # Metrics are resetted every epoch. 
    loss_fn = tf.keras.losses.get(
        config["classification_loss_fn"]
    )(**config["classification_loss_config"])
    
    hiragana_loss_metric = tf.keras.metrics.get(
        config["classification_loss_fn"]
    )(**config["classification_loss_config"])
    katakana_loss_metric = tf.keras.metrics.get(
        config["classification_loss_fn"]
    )(**config["classification_loss_config"])

    hiragana_val_metric = tf.keras.metrics.get(
        config["classification_val_metric"]
    )
    katakana_val_metric = tf.keras.metrics.get(
        config["classification_val_metric"]
    )

    #########################################
    #             TF FUNCTIONS              #
    ######################################### 
    # Wrap the training steps in tf.function for performance

    @tf.function
    def hiragana_classifier_train_step(img, lbl):
        """
        Training Step for Hiragana Classifier 
        """
        with tf.GradientTape() as tape:
            reps = hiragana_classifier(img, training=True)
            y_true, y_pred = get_pred_and_label(reps, lbl)
            loss = loss_fn(y_true, y_pred)

        grads = tape.gradient(
            loss,
            hiragana_classifier.trainable_weights
        )
        hiragana_optimizer.apply_gradients(
            zip(grads, hiragana_classifier.trainable_weights)
        )
        hiragana_loss_metric.update_state(y_true, y_pred)

        return loss
    

    @tf.function
    def katakana_classifier_train_step(img, lbl):
        """
        Training Step for Katakana Classifier 
        """
        with tf.GradientTape() as tape:
            reps = katakana_classifier(img, training=True)
            y_true, y_pred = get_pred_and_label(reps, lbl)
            loss = loss_fn(y_true, y_pred)

        grads = tape.gradient(
            loss,
            katakana_classifier.trainable_weights
        )
        katakana_optimizer.apply_gradients(
            zip(grads, katakana_classifier.trainable_weights)
        )
        katakana_loss_metric.update_state(y_true, y_pred)

        return loss
    

    @tf.function
    def hiragana_classifier_val_step(img, lbl):
        """
        Validation Step for Hiragana Classifier
        """
        reps = hiragana_classifier(img, training=False)
        y_true, y_pred = get_pred_and_label(reps, lbl)
        hiragana_val_metric.update_state(y_true, y_pred)


    @tf.function
    def katakana_classifier_val_step(img, lbl):
        """
        Validation Step for Katakana Classifier
        """
        reps = katakana_classifier(img, training=False)
        y_true, y_pred = get_pred_and_label(reps, lbl)
        katakana_val_metric.update_state(y_true, y_pred)


    #########################################
    #            TRAINING LOOP              #
    ######################################### 
    print(
        f"Beginning classifier training for " 
        f"{config['train']['classifier_training_epochs']} epochs."
    )

    # Train both classifiers at the same time.
    for epoch in range(config["classifier_training_epochs"]):
        print(f"Start of epoch {epoch+1}")

        hiragana_loss_metric.reset_state()
        katakana_loss_metric.reset_state()
        for step, (hiragana_img, katakana_img, label) in enumerate(train_dataset):
            hiragana_loss = hiragana_classifier_train_step(hiragana_img, label)
            katakana_loss = katakana_classifier_train_step(katakana_img, label)
            
            # Log metrics
            if step % 100 == 0:
                print(
                    f"Epoch Loss on hiragana classifier: {hiragana_loss_metric.result():.4f}"
                    " --- "
                    f"Epoch Loss on katakana classifier: {katakana_loss_metric.result():.4f}"
                    " --- "
                    f"Batch Loss on hiragana classifier: {hiragana_loss:.4f}"
                    " --- "
                    f"Batch Loss on katakana classifier: {katakana_loss:.4f}"
                )

        # VALIDATION LOOP
        print("Calculating validation metrics")
        hiragana_val_metric.reset_state()
        katakana_val_metric.reset_state()
        for (hiragana_img, katakana_img, label) in val_dataset:
            hiragana_classifier_val_step(hiragana_img, label)
            katakana_classifier_val_step(katakana_img, label)

        print(
            f"Val Accuracy of hiragana classifier {hiragana_val_metric.result():.4f}"
            " --- "
            f"Val Accuracy of katakana classifier {katakana_val_metric.result():.4f}"
        )

    print("Classifier training complete.")



    

    















