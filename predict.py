"""
Prediction Functions
"""

import os
from typing import Dict

import numpy as np
from PIL import Image
import tensorflow as tf
import yaml

from dataset import preprocessing
from training.utils import get_pred

def predict(config: Dict, model_version: str, filepath: str, kana_type: str, reps: int, loops: bool):
    """
    Run Prediction
    """
    if os.path.isdir(filepath):
        dataset = tf.keras.utils.image_dataset_from_directory(
            filepath,
            labels=None,
            label_mode=None,
            color_mode="grayscale",
            batch_size=1,
            image_size=config["image_size"],
            interpolation="bilinear",
            shuffle=False
        )
    else:
        img = tf.keras.utils.load_img(
            filepath,   
            color_mode="grayscale",
            target_size=config["image_size"],
            interpolation="bilinear"
        )
        img = tf.keras.utils.img_to_array(img)
        dataset = tf.data.Dataset.from_tensors([img])

    dataset = preprocessing(
        dataset,
        crop_image=config["crop_image"],
        predict=kana_type
    )

    if kana_type == "h":
        classifier_path = os.path.join(config["classifier_save_path"], "hiragana", model_version)
        reverse_classifier_path = os.path.join(config["classifier_save_path"], "katakana", model_version)
        generator_path = os.path.join(config["generator_save_path"], "katakana", model_version)
        reverse_generator_path = os.path.join(config["generator_save_path"], "hiragana", model_version)
    else:
        classifier_path = os.path.join(config["classifier_save_path"], "katakana", model_version)
        reverse_classifier_path = os.path.join(config["classifier_save_path"], "hiragana", model_version)
        generator_path = os.path.join(config["generator_save_path"], "hiragana", model_version)
        reverse_generator_path = os.path.join(config["generator_save_path"], "katakana", model_version)
    mapping_path = os.path.join(config["mapping_save_path"], model_version, "mapping.yaml")


    print(f"Loading models")
    classifier = tf.keras.models.load_model(classifier_path)
    generator = tf.keras.models.load_model(generator_path)
    reverse_classifier = tf.keras.models.load_model(reverse_classifier_path)
    reverse_generator = tf.keras.models.load_model(reverse_generator_path)
    with open(mapping_path, "r") as stream:
        label_mapping = yaml.safe_load(stream)
    print("Models loaded")

    for i, img in enumerate(dataset):
        input_img = img.numpy()
        input_img = np.squeeze(input_img)*255
        
        for l in range(loops):
            for r in range(reps):
                pred_rep = classifier(img)
                if r == 0:
                    pred_class = tf.argmax(tf.squeeze(get_pred(pred_rep), axis=0)).numpy()
                    pred_class = label_mapping[pred_class]
                    print(f"Predicted class of input {i}: {pred_class}")
                img = generator(pred_rep)

            reverse_img = img
            for r in range(reps):
                pred_rep = reverse_classifier(reverse_img)
                if r == 0:
                    pred_class = tf.argmax(tf.squeeze(get_pred(pred_rep), axis=0)).numpy()
                    pred_class = label_mapping[pred_class]
                    print(f"Predicted class of generated {i}: {pred_class}")
                reverse_img = reverse_generator(pred_rep)
        
        pred_img = img.numpy()
        reverse_img = reverse_img.numpy()
        pred_img = np.squeeze(pred_img)*255
        reverse_img = np.squeeze(reverse_img)*255
        
        Image.fromarray(input_img).convert('RGB').save(f"input_{i}.png")
        Image.fromarray(pred_img).convert('RGB').save(f"generated_{i}.png")
        Image.fromarray(reverse_img).convert('RGB').save(f"reversed_{i}.png")
