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

def visualize_capsules(config: Dict, model_version: str, filepath: str, kana_type: str, reps: int, loops: bool):
    """
    Visualize the capsule activations on a given input.
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
    else:
        classifier_path = os.path.join(config["classifier_save_path"], "katakana", model_version)
    mapping_path = os.path.join(config["mapping_save_path"], model_version, "mapping.yaml")


    print(f"Loading models")
    classifier = tf.keras.models.load_model(classifier_path)
    with open(mapping_path, "r") as stream:
        label_mapping = yaml.safe_load(stream)
    print("Models loaded")

    for i, img in enumerate(dataset):
        input_img = img.numpy()
        input_img = np.squeeze(input_img)*255
        
        pred_rep = classifier(img)
        pred_class = tf.argmax(tf.squeeze(get_pred(pred_rep), axis=0)).numpy()
        pred_class = label_mapping[pred_class]
        print(f"Predicted class of input {i}: {pred_class}")
        
        pred_rep = tf.squeeze(pred_rep, axis=0).numpy() # num_capsule, dim_capsule

        heatmap_image = (pred_rep.get_array()-pred_rep.get_clim()[0])/(pred_rep.get_clim()[1]-pred_rep.get_clim()[0])
        heatmap_image = Image.fromarray(np.uint8(pred_rep.get_cmap()(heatmap_image) * 255))
        
        Image.fromarray(input_img).convert('RGB').save(f"input_{i}.png")
        Image.fromarray(heatmap_image).convert('RGB').save(f"heatmap_{i}.png")
