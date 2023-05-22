"""
Training the generators based on classifier labels
"""
import os
import time
from typing import Dict

import tensorflow as tf

from ..model.generator import KanaGenerator
from .utils import get_pred_and_label


def train(
    config: Dict,
    hiragana_classifier: tf.keras.Model,
    katakana_classifier: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
):
    """
    Training script
    """
    # Freeze classifier models.
    hiragana_classifier.trainable=False
    katakana_classifier.trainable=False

    # Declare models
    hiragana_generator = KanaGenerator(output_shape=config["image_size"]+[1])
    katakana_generator = KanaGenerator(output_shape=config["image_size"]+[1])

    # Optimizers for generators
    hiragana_to_katakana_optimizer = tf.keras.optimizers.get(
        config["optimizer_type"]
    )(**config["optimizer_config"])
    katakana_to_hiragana_optimizer = tf.keras.optimizers.get(
        config["optimizer_type"]
    )(**config["optimizer_config"])

    #########################################
    #       LOSS FUNCTION AND METRICS       #
    #########################################
    # Metrics are resetted every epoch. 
    classification_loss_fn = tf.keras.losses.get(
        config["classification_loss_fn"]
    )(**config["classification_loss_config"])
    reconstruction_loss_fn = tf.keras.losses.get(
        config["reconstruction_loss_fn"]
    )(**config["reconstruction_loss_config"])

    hiragana_to_katakana_classification_loss_metric = tf.keras.metrics.get(
        config["classification_loss_fn"]
    )(**config["classification_loss_config"])
    katakana_to_hiragana_classification_loss_metric = tf.keras.metrics.get(
        config["classification_loss_fn"]
    )(**config["classification_loss_config"])
    hiragana_to_katakana_reconstruction_loss_metric = tf.keras.metrics.get(
        config["reconstruction_loss_fn"]
    )(**config["reconstruction_loss_config"])
    katakana_to_hiragana_reconstruction_loss_metric = tf.keras.metrics.get(
        config["reconstruction_loss_fn"]
    )(**config["reconstruction_loss_config"])

    hiragana_to_katakana_val_metric = tf.keras.metrics.get(
        config["classification_val_metric"]
    )
    katakana_to_hiragana_val_metric = tf.keras.metrics.get(
        config["classification_val_metric"]
    )

    #########################################
    #             TF FUNCTIONS              #
    ######################################### 
    # Wrap the training steps in tf.function for performance

    @tf.function
    def hiragana_to_katakana_train_step(hira_img, lbl):
        """
        Train Step for Hiragana -> Katakana
        """
        with tf.GradientTape() as tape:
            # Classification of the generated sample.
            hiragana_reps = hiragana_classifier(hira_img, training=False)
            katakana_gen = katakana_generator(hiragana_reps, training=True)
            katakana_pred = katakana_classifier(katakana_gen, training=False)
            y_true, y_pred = get_pred_and_label(katakana_pred, lbl)
            classification_loss = classification_loss_fn(y_true, y_pred)

            # Reconstruction
            hiragana_recon = hiragana_generator(katakana_pred, training=True)
            reconstruction_loss = reconstruction_loss_fn(hira_img, hiragana_recon)
            
            total_loss = classification_loss + reconstruction_loss
        
        grads = tape.gradient(
            total_loss,
            ( # list concatenation of weights
                katakana_generator.trainable_weights,
                + hiragana_generator.trainable_weights
            )
        )
        hiragana_to_katakana_optimizer.apply_gradients(
            zip(
                grads,
                ( # list concatenation of weights
                    katakana_generator.trainable_weights,
                    + hiragana_generator.trainable_weights
                )
            )
        )

        hiragana_to_katakana_classification_loss_metric.update_state(y_true, y_pred)
        hiragana_to_katakana_reconstruction_loss_metric.update_state(hiragana_img, hiragana_recon)

        return classification_loss, reconstruction_loss


    @tf.function
    def katakana_to_hiragana_train_step(kata_img, lbl):
        """
        Train Step for Katakana -> Hiragana
        """
        with tf.GradientTape() as tape:
            # Classification of the generated sample.
            katakana_reps = katakana_classifier(kata_img, training=False)
            hiragana_gen = hiragana_generator(katakana_reps, training=True)
            hiragana_pred = hiragana_classifier(hiragana_gen, training=False)
            y_true, y_pred = get_pred_and_label(hiragana_pred, lbl)
            classification_loss = classification_loss_fn(y_true, y_pred)

            # Reconstruction
            katakana_recon = katakana_generator(hiragana_pred, training=True)
            reconstruction_loss = reconstruction_loss_fn(kata_img, katakana_recon)
            
            total_loss = classification_loss + reconstruction_loss
        
        grads = tape.gradient(
            total_loss,
            ( # list concatenation of weights
                hiragana_generator.trainable_weights,
                + katakana_generator.trainable_weights
            )
        )
        katakana_to_hiragana_optimizer.apply_gradients(
            zip(
                grads,
                ( # list concatenation of weights
                    hiragana_generator.trainable_weights,
                    + katakana_generator.trainable_weights
                )
            )
        )

        katakana_to_hiragana_classification_loss_metric.update_state(y_true, y_pred)
        katakana_to_hiragana_reconstruction_loss_metric.update_state(kata_img, katakana_recon)

        return classification_loss, reconstruction_loss
    

    @tf.function
    def hiragana_to_katakana_val_step(hira_img, lbl):
        """
        Validation Step for Hiragana -> Katakana
        """
        hiragana_reps = hiragana_classifier(hira_img, training=False)
        katakana_gen = katakana_generator(hiragana_reps, training=False)
        katakana_pred = katakana_classifier(katakana_gen, training=False)
        y_true, y_pred = get_pred_and_label(katakana_pred, lbl)
        hiragana_to_katakana_val_metric.update_state(y_true, y_pred)


    @tf.function
    def katakana_to_hiragana_val_step(kata_img, lbl):
        """
        Validation Step for Katakana -> Hiragana
        """
        katakana_reps = katakana_classifier(kata_img, training=False)
        hiragana_gen = hiragana_generator(katakana_reps, training=False)
        hiragana_pred = hiragana_classifier(hiragana_gen, training=False)
        y_true, y_pred = get_pred_and_label(hiragana_pred, lbl)
        katakana_to_hiragana_val_metric.update_state(y_true, y_pred)


    #########################################
    #          GENERATOR TRAINING           #
    #########################################
    t0 = last_ckpt_time = time.localtime()
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S', t0)}] "
        "Beginning generator training for " 
        f"{config['train']['classifier_training_epochs']} epochs."
    )

    # Initialize checkpointing
    hiragana_to_katakana_ckpt = tf.train.Checkpoint(
        optimizer=hiragana_to_katakana_optimizer,
        model=katakana_generator
    )
    hiragana_to_katakana_mgr = tf.train.CheckpointManager(
        checkpoint=hiragana_to_katakana_ckpt,
        directory=os.path.join(config["checkpoint_dir"], "generator", "katakana"),
        max_to_keep=1
    )

    katakana_to_hiragana_ckpt = tf.train.Checkpoint(
        optimizer=katakana_to_hiragana_optimizer,
        model=hiragana_generator
    )
    katakana_to_hiragana_mgr = tf.train.CheckpointManager(
        checkpoint=katakana_to_hiragana_ckpt,
        directory=os.path.join(config["checkpoint_dir"], "generator", "hiragana"),
        max_to_keep=1
    )

    if hiragana_to_katakana_mgr.latest_checkpoint:
        hiragana_to_katakana_ckpt.restore(hiragana_to_katakana_mgr.latest_checkpoint)
    if katakana_to_hiragana_mgr.latest_checkpoint:
        katakana_to_hiragana_ckpt.restore(katakana_to_hiragana_mgr.latest_checkpoint)

    # Dual training scheme
    for epoch in range(config["generator_training_epochs"]):
        print(f"Start of epoch {epoch+1}")

        hiragana_to_katakana_classification_loss_metric.reset_state()
        katakana_to_hiragana_classification_loss_metric.reset_state()
        hiragana_to_katakana_reconstruction_loss_metric.reset_state()
        katakana_to_hiragana_reconstruction_loss_metric.reset_state()
        for step, (hiragana_img, katakana_img, label) in enumerate(train_dataset):
            (
                hiragana_to_katakana_classification_loss,
                hiragana_to_katakana_reconstruction_loss,
            ) = hiragana_to_katakana_train_step(hiragana_img, label)
            (
                katakana_to_hiragana_classification_loss,
                katakana_to_hiragana_reconstruction_loss,
            ) = katakana_to_hiragana_train_step(katakana_img, label)

            # Log metrics
            if step % 100 == 0:
                print(
                    "Hiragana to Katakana: \n"
                    f"Epoch Classification Loss: {hiragana_to_katakana_classification_loss_metric.result():.4f}"
                    " --- "
                    f"Epoch Reconstruction Loss: {hiragana_to_katakana_reconstruction_loss_metric.result():.4f}"
                    " --- "
                    f"Batch Classification Loss: {hiragana_to_katakana_classification_loss:.4f}"
                    " --- "
                    f"Batch Reconstruction Loss: {hiragana_to_katakana_reconstruction_loss:.4f}"
                )
                print(
                    "Katakana to Hiragana: \n"
                    f"Epoch Classification Loss: {katakana_to_hiragana_classification_loss_metric.result():.4f}"
                    " --- "
                    f"Epoch Reconstruction Loss: {katakana_to_hiragana_reconstruction_loss_metric.result():.4f}"
                    " --- "
                    f"Batch Classification Loss: {katakana_to_hiragana_classification_loss:.4f}"
                    " --- "
                    f"Batch Reconstruction Loss: {katakana_to_hiragana_reconstruction_loss:.4f}"
                )

        # VALIDATION LOOP
        print("Calculating validation metrics")

        hiragana_to_katakana_val_metric.reset_state()
        katakana_to_hiragana_val_metric.reset_state()
        for (hiragana_img, katakana_img, label) in val_dataset:
            hiragana_to_katakana_val_step(hiragana_img, label)
            katakana_to_hiragana_val_step(katakana_img, label)

        print(
            f"Val Accuracy on generated Katakana images {hiragana_to_katakana_val_metric.result():.4f}"
            " --- "
            f"Val Accuracy on generated Hiragana images {katakana_to_hiragana_val_metric.result():.4f}"
        )

        # CHECKPOINTING
        if (time.mktime(time.localtime()) - time.mktime(last_ckpt_time)) >= config["checkpoint_interval"]:
            last_ckpt_time = time.localtime()
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S', last_ckpt_time)}] "
                f"Saving checkpoint at epoch {epoch}" 
            )
            hiragana_to_katakana_mgr.save()
            katakana_to_hiragana_mgr.save()

    print("Training Complete")

    return hiragana_classifier, katakana_classifier, hiragana_generator, katakana_generator