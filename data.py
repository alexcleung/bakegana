"""
Dataset processing functions
"""

import os
from typing import List

import tensorflow as tf

from .constants import CHAR_MAPPINGS


def validate_subdirectories(data_dir: str, character: str):
    """
    Check that the subdirectories (for both the hiragana and the katakana)
    for a given character exists.
    """
    char_unicodes = CHAR_MAPPINGS.get(character)
    if not char_unicodes:
        raise ValueError(f"Could not find any mapping for character {character}")
    
    hiragana_unicode, katakana_unicode = char_unicodes

    hiragana_images = os.listdir(os.path.join(data_dir, hiragana_unicode))
    if len(hiragana_images) == 0:
        raise ValueError(f"No Hiragana images for character {character}")
    katakana_images = os.listdir(os.path.join(data_dir, katakana_unicode))
    if len(katakana_images) == 0:
        raise ValueError(f"No Katakana images for character {character}")
    
    print(
        f"Found {len(hiragana_images)} hiragana images"
        f" and {len(katakana_images)} katakana_images" 
        f" for character {character}"
    )


def preprocessing(dataset, contrast_factor: float):
    """
    Apply preprocessing transformations to the dataset.
    """
    # Invert - white characters on black background
    dataset = dataset.map(
        lambda x: (1-x[0], 1-x[1], x[2]),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Increase contrast
    dataset = dataset.map(
        lambda x: (
            tf.image.adjust_constrast(x[0], contrast_factor),
            tf.image.adjust_constrast(x[1], contrast_factor),
            x[2]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return dataset


def create_dataset(
        data_dir: str, 
        characters: List[str],
        image_size: List[int],
        batch_size: int,
        contrast_factor: float = 3.0,
    ):
    """
    Create a TF Dataset of images of kana `characters`.

    `data_dir`: Top level directory containing a subdirectory for each
        hiragana/katakana character. The subdirectories are named using
        the unicode for each character.
    `characters`: the romanized character for each kana to be included in
        in the dataset.
    `batch_size`: batch size of the dataset
    `contrast_factor`: contrast to apply on preprocessing.

    Returns: 
        2 TF Datasets (train and val) that each yield 3 elements:
            `hiragana_image`: Tensor of shape [batch, width, height, channels]
            `katakana_image`: Tensor of shape [batch, width, height, channels]
            `labels`: Tensor of shape [batch, 1] containing integer labels

        label mapping: Dictionary mapping integer label to string character.
    """
    characters = set(characters)

    for c in characters:
        validate_subdirectories(data_dir, c)

    # Create a train/val dataset for each character,
    # Maintain the mapping between integer index label to character.
    datasets = []
    label_mapping = {}
    for i, c in enumerate(characters):
        hiragana_dir = os.path.join(data_dir, CHAR_MAPPINGS[c][0])
        katakana_dir = os.path.join(data_dir, CHAR_MAPPINGS[c][0])

        hiragana_train, hiragana_val = tf.keras.utils.image_dataset_from_directory(
            directory=hiragana_dir,
            labels=[i] * len(os.listdir(hiragana_dir)),
            label_mode=None,
            color_mode="grayscale",
            image_size=image_size,
            shuffle=False,
            validation_split=0.1,
            subset="both"
        )

        katakana_train, katakana_val = tf.keras.utils.image_dataset_from_directory(
            directory=katakana_dir,
            labels=[i] * len(os.listdir(katakana_dir)),
            label_mode="int",
            color_mode="grayscale",
            image_size=image_size,
            shuffle=False,
            validation_split=0.1,
            subset="both"
        )

        train = tf.data.Dataset.zip((hiragana_train, katakana_train))
        val = tf.data.Dataset.zip((hiragana_val, katakana_val))

        # Only the katakana datasets return labels; use map to flatten.
        # so that datasets yield (hiragana_img, katakana_img, label)
        train = train.map(lambda x,y: (x, *y), num_parallel_calls=tf.data.AUTOTUNE)
        val = val.map(lambda x,y: (x, *y), num_parallel_calls=tf.data.AUTOTUNE)

        datasets.append((train, val))
        label_mapping[i] = c
        
    # Concatenate all the datasets together.
    train_ds, val_ds = datasets.pop(0)
    while datasets:
        next_train_ds, next_val_ds = datasets.pop(0)
        train_ds.concatenate(next_train_ds)
        val_ds.concatenate(next_val_ds)
    
    # Shuffle so that characters are in random order
    train_ds = train_ds.shuffle(buffer_size=20000, reshuffle_each_iteration=True)
    val_ds = val_ds.shuffle(buffer_size=20000, reshuffle_each_iteration=True)

    # Batch
    train_ds = train_ds.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)

    # Preprocess
    train_ds = preprocessing(train_ds, contrast_factor=contrast_factor)
    val_ds = preprocessing(val_ds, contrast_factor=contrast_factor)

    return train_ds, val_ds, label_mapping
