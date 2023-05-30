"""
Training the generators based on classifier labels
"""
import os
import time
from typing import Dict

import tensorflow as tf

from model.generator import KanaGenerator2 as KanaGenerator
from .utils import get_true_and_pred, apply_training_mask


def train(
    config: Dict,
    hiragana_classifier: tf.keras.Model,
    katakana_classifier: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    timestamp: str
):
    """
    Training script
    """
    # Freeze classifier models.
    hiragana_classifier.trainable=False
    katakana_classifier.trainable=False

    # Declare models
    hiragana_generator = KanaGenerator(
        image_shape=([52,36] if config["crop_image"] else config["image_size"])+[1]
        )
    katakana_generator = KanaGenerator(
        image_shape=([52,36] if config["crop_image"] else config["image_size"])+[1]
    )

    # Optimizers for generators
    optimizer = tf.keras.optimizers.Adam(
        **config["optimizer_config"]
    )

    #########################################
    #       LOSS FUNCTION AND METRICS       #
    #########################################
    # Metrics are resetted every epoch. 
    classification_loss_fn = tf.keras.losses.get(
        {
            "class_name": config["classification_loss_fn"],
            "config": config["classification_loss_config"]
        }
    )
    reconstruction_loss_fn = tf.keras.losses.get(
        {
            "class_name": config["reconstruction_loss_fn"],
            "config": config["reconstruction_loss_config"]
        }
    )
    reconstruction_loss_coef = config["reconstruction_loss_coef"]

    sample_weight_scaling_fn = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE
    )

    hiragana_to_katakana_classification_loss_metric = tf.keras.metrics.get(
        {
            "class_name": config["classification_loss_fn"],
            "config": config["classification_loss_config"]
        }
    )
    katakana_to_hiragana_classification_loss_metric = tf.keras.metrics.get(
        {
            "class_name": config["classification_loss_fn"],
            "config": config["classification_loss_config"]
        }
    )
    hiragana_to_katakana_reconstruction_loss_metric = tf.keras.metrics.get(
        {
            "class_name": config["reconstruction_loss_fn"],
            "config": config["reconstruction_loss_config"]
        }
    )
    katakana_to_hiragana_reconstruction_loss_metric = tf.keras.metrics.get(
        {
            "class_name": config["reconstruction_loss_fn"],
            "config": config["reconstruction_loss_config"]
        }
    )
    
    hiragana_to_katakana_val_metric = tf.keras.metrics.get(
        config["classification_val_metric"]
    )
    best_result_hiragana_to_katakana_val_metric = 0

    katakana_to_hiragana_val_metric = tf.keras.metrics.get(
        config["classification_val_metric"]
    )
    best_result_katakana_to_hiragana_val_metric = 0

    #########################################
    #             TF FUNCTIONS              #
    ######################################### 
    # Wrap the training steps in tf.function for performance

    mask_during_training = config["mask_during_training"]

    @tf.function
    def hiragana_to_katakana_train_step(hira_img, lbl, recon_loss_coef, mask):
        """
        Train Step for Hiragana -> Katakana
        """
        with tf.GradientTape() as tape:
            # Classification of the generated sample.
            hiragana_reps = hiragana_classifier(hira_img, training=False)
            hira_true, hira_pred = get_true_and_pred(hiragana_reps, lbl)
            classification_sample_weight = tf.exp(-sample_weight_scaling_fn(hira_true, hira_pred))
            if mask:
                hiragana_reps = apply_training_mask(hiragana_reps, lbl)
            katakana_gen = katakana_generator(hiragana_reps, training=True)
            katakana_pred = katakana_classifier(katakana_gen, training=False)
            y_true, y_pred = get_true_and_pred(katakana_pred, lbl)
            classification_loss = classification_loss_fn(
                y_true,
                y_pred,
                sample_weight=classification_sample_weight[:, tf.newaxis]
            )

            # Reconstruction
            reconstruction_sample_weight = tf.exp(-sample_weight_scaling_fn(y_true, y_pred))
            if mask:
                katakana_pred = apply_training_mask(katakana_pred, lbl)
            katakana_pred = tf.stop_gradient(katakana_pred) # reconstruction loss should not backprop to hira generator
            hiragana_recon = hiragana_generator(katakana_pred, training=True)
            reconstruction_loss = reconstruction_loss_fn(
                hira_img,
                hiragana_recon,
                sample_weight=reconstruction_sample_weight[:, tf.newaxis, tf.newaxis]
            ) * recon_loss_coef
            
            total_loss = classification_loss + reconstruction_loss
        
        grads = tape.gradient(
            total_loss,
            ( # list concatenation of weights
                katakana_generator.trainable_weights
                + hiragana_generator.trainable_weights
            )
        )
        optimizer.apply_gradients(
            zip(
                grads,
                ( # list concatenation of weights
                    katakana_generator.trainable_weights
                    + hiragana_generator.trainable_weights
                )
            )
        )

        hiragana_to_katakana_classification_loss_metric.update_state(y_true, y_pred)
        hiragana_to_katakana_reconstruction_loss_metric.update_state(hiragana_img, hiragana_recon)

        return classification_loss, reconstruction_loss


    @tf.function
    def katakana_to_hiragana_train_step(kata_img, lbl, recon_loss_coef, mask):
        """
        Train Step for Katakana -> Hiragana
        """
        with tf.GradientTape() as tape:
            # Classification of the generated sample.
            katakana_reps = katakana_classifier(kata_img, training=False)
            kata_true, kata_pred = get_true_and_pred(katakana_reps, lbl)
            classification_sample_weight = tf.exp(-sample_weight_scaling_fn(kata_true, kata_pred))
            if mask:
                katakana_reps = apply_training_mask(katakana_reps, lbl)
            hiragana_gen = hiragana_generator(katakana_reps, training=True)
            hiragana_pred = hiragana_classifier(hiragana_gen, training=False)
            y_true, y_pred = get_true_and_pred(hiragana_pred, lbl)
            classification_loss = classification_loss_fn(
                y_true,
                y_pred,
                sample_weight=classification_sample_weight[:, tf.newaxis]
            )

            # Reconstruction
            reconstruction_sample_weight = tf.exp(-sample_weight_scaling_fn(y_true, y_pred))
            if mask:
                hiragana_pred = apply_training_mask(hiragana_pred, lbl)
            hiragana_pred = tf.stop_gradient(hiragana_pred) # reconstruction loss should not backprop to kata generator
            katakana_recon = katakana_generator(hiragana_pred, training=True)
            reconstruction_loss = reconstruction_loss_fn(
                kata_img,
                katakana_recon,
                sample_weight=reconstruction_sample_weight[:, tf.newaxis, tf.newaxis]
            ) * recon_loss_coef
            
            total_loss = classification_loss + reconstruction_loss
        
        grads = tape.gradient(
            total_loss,
            ( # list concatenation of weights
                hiragana_generator.trainable_weights
                + katakana_generator.trainable_weights
            )
        )
        optimizer.apply_gradients(
            zip(
                grads,
                ( # list concatenation of weights
                    hiragana_generator.trainable_weights
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
        y_true, y_pred = get_true_and_pred(katakana_pred, lbl)
        hiragana_to_katakana_val_metric.update_state(y_true, y_pred)


    @tf.function
    def katakana_to_hiragana_val_step(kata_img, lbl):
        """
        Validation Step for Katakana -> Hiragana
        """
        katakana_reps = katakana_classifier(kata_img, training=False)
        hiragana_gen = hiragana_generator(katakana_reps, training=False)
        hiragana_pred = hiragana_classifier(hiragana_gen, training=False)
        y_true, y_pred = get_true_and_pred(hiragana_pred, lbl)
        katakana_to_hiragana_val_metric.update_state(y_true, y_pred)


    #########################################
    #          GENERATOR TRAINING           #
    #########################################
    t0 = time.localtime()
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S', t0)}] "
        "Beginning generator training for " 
        f"{config['classifier_training_epochs']} epochs."
    )

    # Dual training scheme
    epochs_since_improvement = 0
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
            ) = hiragana_to_katakana_train_step(
                    hiragana_img,
                    label,
                    reconstruction_loss_coef,
                    mask_during_training
                )
            (
                katakana_to_hiragana_classification_loss,
                katakana_to_hiragana_reconstruction_loss,
            ) = katakana_to_hiragana_train_step(
                    katakana_img,
                    label,
                    reconstruction_loss_coef,
                    mask_during_training
                )

            # Log metrics
            if step % 10 == 0:
                print(
                    "Hiragana to Katakana: \n"
                    f"Epoch Classification Loss: {hiragana_to_katakana_classification_loss_metric.result():.4f}"
                    " --- "
                    f"Epoch Reconstruction Loss: {hiragana_to_katakana_reconstruction_loss_metric.result() * reconstruction_loss_coef:.4f}"
                    " --- "
                    f"Batch Classification Loss: {hiragana_to_katakana_classification_loss:.4f}"
                    " --- "
                    f"Batch Reconstruction Loss: {hiragana_to_katakana_reconstruction_loss * reconstruction_loss_coef:.4f}"
                )
                print(
                    "Katakana to Hiragana: \n"
                    f"Epoch Classification Loss: {katakana_to_hiragana_classification_loss_metric.result():.4f}"
                    " --- "
                    f"Epoch Reconstruction Loss: {katakana_to_hiragana_reconstruction_loss_metric.result() * reconstruction_loss_coef:.4f}"
                    " --- "
                    f"Batch Classification Loss: {katakana_to_hiragana_classification_loss:.4f}"
                    " --- "
                    f"Batch Reconstruction Loss: {katakana_to_hiragana_reconstruction_loss * reconstruction_loss_coef:.4f}"
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

        # PLATEAU CONDITIONS
        if (
            (hiragana_to_katakana_val_metric.result() > best_result_hiragana_to_katakana_val_metric)
            or (katakana_to_hiragana_val_metric.result() > best_result_katakana_to_hiragana_val_metric)
        ):
            best_result_hiragana_to_katakana_val_metric = max(hiragana_to_katakana_val_metric.result(), best_result_hiragana_to_katakana_val_metric)
            best_result_katakana_to_hiragana_val_metric = max(katakana_to_hiragana_val_metric.result(), best_result_katakana_to_hiragana_val_metric)
            epochs_since_improvement = 0
        else:
            epochs_since_improvement +=1

        if epochs_since_improvement >= config["reduce_lr_epochs_since_improvement"]:
            print("HALVING LEARNING RATE")
            optimizer.lr.assign(optimizer.lr/2)

        # if epochs_since_improvement >= config["early_stopping_epochs_since_improvement"]:
        #     print("EARLY STOPPING")
        #     break

    print("Training Complete")

    #########################################
    #              SAVE MODELS              #
    #########################################

    print("Saving models")

    hiragana_generator.save(
        os.path.join(config["generator_save_path"], "hiragana", timestamp)
    )

    katakana_generator.save(
        os.path.join(config["generator_save_path"], "katakana", timestamp)
    )

    return hiragana_classifier, katakana_classifier, hiragana_generator, katakana_generator