"""
Training the classifiers based on ground truth
"""
import os
import time
from typing import Dict

import tensorflow as tf
import yaml

from model.classifier import KanaClassifier
from model.generator import KanaGenerator
from .utils import get_true_and_pred, apply_training_mask


def train(
    config: Dict,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    label_mapping: Dict,
    timestamp: str
):
    """
    Training script
    """
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # Declare models
    hiragana_classifier = KanaClassifier(
        n_classes=len(label_mapping),
        dim_output=config["embedding_dim"],
        n_routings=config["n_routings"],
        capsule_l2=config["capsule_l2"]
    )
    katakana_classifier = KanaClassifier(
        n_classes=len(label_mapping),
        dim_output=config["embedding_dim"],
        n_routings=config["n_routings"],
        capsule_l2=config["capsule_l2"]
    )
    hiragana_generator = KanaGenerator(
        image_shape=([52,36] if config["crop_image"] else config["image_size"])+[1]
        )
    katakana_generator = KanaGenerator(
        image_shape=([52,36] if config["crop_image"] else config["image_size"])+[1]
    )

    # Optimizers for classifiers
    hiragana_optimizer = tf.keras.optimizers.Adam(
        **config["optimizer_config"]
    )
    katakana_optimizer = tf.keras.optimizers.Adam(
        **config["optimizer_config"]
    )

    #########################################
    #       LOSS FUNCTION AND METRICS       #
    #########################################
    # Metrics are resetted every epoch. 
    class_loss_fn = tf.keras.losses.get(
        {
            "class_name": config["classification_loss_fn"],
            "config": config["classification_loss_config"]
        }
    )
    recon_loss_fn = tf.keras.losses.get(
        {
            "class_name": config["reconstruction_loss_fn"],
            "config": config["reconstruction_loss_config"]
        }
    )
    recon_reg_coef = config["reconstruction_reg_coef"]

    hiragana_loss_metric = tf.keras.metrics.get(
        {
            "class_name": config["classification_loss_fn"],
            "config": config["classification_loss_config"]
        }
    )
    katakana_loss_metric = tf.keras.metrics.get(
        {
            "class_name": config["classification_loss_fn"],
            "config": config["classification_loss_config"]
        }
    )

    hiragana_val_metric = tf.keras.metrics.get(
        config["classification_val_metric"]
    )
    best_result_hiragana_val_metric = 0

    katakana_val_metric = tf.keras.metrics.get(
        config["classification_val_metric"]
    )
    best_result_katakana_val_metric = 0

    #########################################
    #             TF FUNCTIONS              #
    ######################################### 
    # Wrap the training steps in tf.function for performance

    @tf.function
    def hiragana_classifier_train_step(img, lbl, recon_coef):
        """
        Training Step for Hiragana Classifier 
        """
        with tf.GradientTape() as tape:
            reps = hiragana_classifier(img, training=True)
            y_true, y_pred = get_true_and_pred(reps, lbl)
            reps = apply_training_mask(reps, lbl)
            recon = hiragana_generator(reps)
            loss = (
                class_loss_fn(y_true, y_pred)
                + sum(hiragana_classifier.losses) # reg loss
                + recon_loss_fn(img, recon) * recon_coef
            )

        grads = tape.gradient(
            loss,
            ( # list concat
                hiragana_classifier.trainable_weights
                + hiragana_generator.trainable_weights
            )
        )
        hiragana_optimizer.apply_gradients(
            zip(grads, hiragana_classifier.trainable_weights)
        )
        hiragana_loss_metric.update_state(y_true, y_pred)

        return loss
    

    @tf.function
    def katakana_classifier_train_step(img, lbl, recon_coef):
        """
        Training Step for Katakana Classifier 
        """
        with tf.GradientTape() as tape:
            reps = katakana_classifier(img, training=True)
            y_true, y_pred = get_true_and_pred(reps, lbl)
            reps = apply_training_mask(reps, lbl)
            recon = katakana_generator(reps)
            loss = (
                class_loss_fn(y_true, y_pred)
                + sum(katakana_classifier.losses) # reg loss
                + recon_loss_fn(img, recon) * recon_coef
            )

        grads = tape.gradient(
            loss,
            ( # list concat
                katakana_classifier.trainable_weights
                + katakana_generator.trainable_weights
            )
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
        y_true, y_pred = get_true_and_pred(reps, lbl)
        hiragana_val_metric.update_state(y_true, y_pred)


    @tf.function
    def katakana_classifier_val_step(img, lbl):
        """
        Validation Step for Katakana Classifier
        """
        reps = katakana_classifier(img, training=False)
        y_true, y_pred = get_true_and_pred(reps, lbl)
        katakana_val_metric.update_state(y_true, y_pred)


    #########################################
    #            TRAINING LOOP              #
    ######################################### 
    t0 = time.localtime()
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S', t0)}] "
        "Beginning classifier training for " 
        f"{config['classifier_training_epochs']} epochs."
    )

    # Train both classifiers at the same time.
    hiragana_epochs_since_improvement = 0
    katakana_epochs_since_improvement = 0
    train_hiragana = True
    train_katakana = True
    for epoch in range(config["classifier_training_epochs"]):
        print(f"Start of epoch {epoch+1}")

        if train_hiragana:
            hiragana_loss_metric.reset_state()
        if train_katakana:
            katakana_loss_metric.reset_state()
        for step, (hiragana_img, katakana_img, label) in enumerate(train_dataset):
            if train_hiragana:
                hiragana_loss = hiragana_classifier_train_step(hiragana_img, label, recon_reg_coef)
            if train_katakana:
                katakana_loss = katakana_classifier_train_step(katakana_img, label, recon_reg_coef)
            
            # Log metrics
            if step % 10 == 0:
                if train_hiragana:
                    print(
                        f"Epoch Loss on hiragana classifier: {hiragana_loss_metric.result():.4f}"
                        " --- "
                        f"Batch Loss on hiragana classifier: {hiragana_loss:.4f}"
                    )
                if train_katakana:
                    print(
                        f"Epoch Loss on katakana classifier: {katakana_loss_metric.result():.4f}"
                        " --- "
                        f"Batch Loss on katakana classifier: {katakana_loss:.4f}"
                    )

        # VALIDATION LOOP
        print("Calculating validation metrics")
        if train_hiragana:
            hiragana_val_metric.reset_state()
        if train_katakana:
            katakana_val_metric.reset_state()

        for (hiragana_img, katakana_img, label) in val_dataset:
            if train_hiragana:
                hiragana_classifier_val_step(hiragana_img, label)
            if train_katakana:
                katakana_classifier_val_step(katakana_img, label)

        if train_hiragana:
            print(
                f"Val Accuracy of hiragana classifier {hiragana_val_metric.result():.4f}"
            )
        if train_katakana:
            print(
                f"Val Accuracy of katakana classifier {katakana_val_metric.result():.4f}"
            )

        # PLATEAU CONDITIONS
        if (hiragana_val_metric.result() > best_result_hiragana_val_metric):
            best_result_hiragana_val_metric = max(hiragana_val_metric.result(), best_result_hiragana_val_metric)
            hiragana_epochs_since_improvement = 0
        else:
            hiragana_epochs_since_improvement += 1

        if (katakana_val_metric.result() > best_result_katakana_val_metric):
            best_result_katakana_val_metric = max(katakana_val_metric.result(), best_result_katakana_val_metric)
            katakana_epochs_since_improvement = 0
        else:
            katakana_epochs_since_improvement += 1

        if hiragana_epochs_since_improvement == config["reduce_lr_epochs_since_improvement"]:
            print("HALVING LEARNING RATE FOR HIRAGANA CLASSIFIER")
            hiragana_optimizer.lr.assign(hiragana_optimizer.lr/2)
        if katakana_epochs_since_improvement == config["reduce_lr_epochs_since_improvement"]:
            print("HALVING LEARNING RATE FOR KATAKANA CLASSIFIER")
            katakana_optimizer.lr.assign(katakana_optimizer.lr/2)

        if hiragana_epochs_since_improvement == config["early_stopping_epochs_since_improvement"]:
            print("STOPPING TRAINING OF HIRAGANA CLASSIFIER")
            train_hiragana = False
        if katakana_epochs_since_improvement == config["early_stopping_epochs_since_improvement"]:
            print("STOPPING TRAINING OF KATAKANA CLASSIFIER")
            train_katakana = False

        if not (train_hiragana or train_katakana):
            break

    print("Classifier training complete.")

    #########################################
    #              SAVE MODELS              #
    #########################################

    print("Saving models")

    hiragana_classifier.save(
        os.path.join(config["classifier_save_path"], "hiragana", timestamp)
    )

    katakana_classifier.save(
        os.path.join(config["classifier_save_path"], "katakana", timestamp)
    )

    mapping_save_dir = os.path.join(config["mapping_save_path"], timestamp)
    if not os.path.exists(mapping_save_dir):
        os.makedirs(mapping_save_dir)
    with open(os.path.join(mapping_save_dir, "mapping.yaml"), "w") as stream:
        yaml.dump(label_mapping, stream, default_flow_style=False)

    config_save_dir = os.path.join(config["config_save_path"], timestamp)
    if not os.path.exists(config_save_dir):
        os.makedirs(config_save_dir)
    with open(os.path.join(config_save_dir, "config.yaml"), "w") as stream:
        yaml.dump(config, stream, default_flow_style=False)

    return hiragana_classifier, katakana_classifier
