"""
Dataset processing functions
"""
import glob
import os
from typing import Dict

import tensorflow as tf

from constants import CHAR_MAPPINGS


def lower_unicode_hexadecimal(unicode):
    """
    Directories loaded from dataset has hexadecimal digits as lowercase
    Adjust the mapping (uppercase) so that the directories can be found.
    """
    return "".join([c.lower() for c in unicode])


def validate_subdirectories(data_dir: str, character: str):
    """
    Check that the subdirectories (for both the hiragana and the katakana)
    for a given character exists.
    """
    char_unicodes = CHAR_MAPPINGS.get(character)
    if not char_unicodes:
        raise ValueError(f"Could not find any mapping for character {character}")
    
    hiragana_unicode, katakana_unicode = char_unicodes
    hiragana_unicode = lower_unicode_hexadecimal(hiragana_unicode)
    katakana_unicode = lower_unicode_hexadecimal(katakana_unicode)

    hiragana_images = glob.glob(os.path.join(data_dir, hiragana_unicode, "*.png"))
    if len(hiragana_images) == 0:
        raise ValueError(f"No Hiragana images for character {character}")
    katakana_images = glob.glob(os.path.join(data_dir, katakana_unicode, "*.png"))
    if len(katakana_images) == 0:
        raise ValueError(f"No Katakana images for character {character}")
    
    print(
        f"Found {len(hiragana_images)} hiragana images"
        f" and {len(katakana_images)} katakana_images" 
        f" for character {character}"
    )


@tf.function
def remove_noise(img, mean=False):
    """
    Remove image noise by applying mean filter
    `img` is a Tensor of shape [batch, width, height, channels]
    Returns a Tensor of same shape.
    """
    img_shape = tf.shape(img)

    patches = tf.image.extract_patches(
        img,
        sizes=[1, 3, 3, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding="SAME"
    )
    patches = tf.reshape(patches, tf.concat([img_shape, [-1]], axis=0))

    if mean:
        patches = tf.reduce_mean(patches, axis=-1, keepdims=True)
    else:
        patches = tf.sort(patches, axis=-1)
        median_idx = tf.math.ceil(img_shape[-1] / 2)
        patches = tf.slice(
            patches,
            begin=tf.cast(tf.concat([[0,0,0,0], median_idx[tf.newaxis]], axis=0), tf.int32),
            size=tf.concat([img_shape, [1]], axis=0)
        )

    return tf.squeeze(patches, axis=-1)


def preprocessing(dataset, training=True):
    """
    Apply preprocessing transformations to the dataset.
    """

    # Invert - white characters on black background
    dataset = dataset.map(
        lambda *t: 
            (-1*t[0]+255,-1*t[1]+255, t[2]) if training
            else -1*t[0]+255,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.map(
        lambda *t:
            (remove_noise(t[0]), remove_noise(t[1]), t[2]) if training
            else remove_noise(t[0]),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Increase contrast
    dataset = dataset.map(
        lambda *t: 
            (
                tf.image.adjust_contrast(t[0], 4),
                tf.image.adjust_contrast(t[1], 4),
                t[2]
            ) if training
            else tf.image.adjust_contrast(t[0], 4),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # scale to 0 - 1
    dataset = dataset.map(
        lambda *t:
            (t[0]/255, t[1]/255, t[2]) if training
            else t[0]/255,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    return dataset


def create_dataset(
    config: Dict
):
    """
    Create a TF Dataset of images of kana `characters`.

    `config` is a dictionary containing:
        `data_dir`: Top level directory containing a subdirectory for each
            hiragana/katakana character. The subdirectories are named using
            the unicode for each character.
        `characters`: the romanized character for each kana to be included in
            in the dataset.
        `batch_size`: batch size of the dataset

    Returns: 
        2 TF Datasets (train and val) that each yield 3 elements:
            `hiragana_image`: Tensor of shape [batch, width, height, channels]
            `katakana_image`: Tensor of shape [batch, width, height, channels]
            `labels`: Tensor of shape [batch, 1] containing integer labels

        label mapping: Dictionary mapping integer label to string character.
    """
    data_dir = config["data_dir"]
    characters = config["characters"]
    image_size = config["image_size"]
    batch_size = config["batch_size"]

    for c in characters:
        validate_subdirectories(data_dir, c)

    # Create a train/val dataset for each character,
    # Maintain the mapping between integer index label to character.
    datasets = []
    label_mapping = {}
    for i, c in enumerate(characters):
        hiragana_dir = os.path.join(data_dir, lower_unicode_hexadecimal(CHAR_MAPPINGS[c][0]))
        katakana_dir = os.path.join(data_dir, lower_unicode_hexadecimal(CHAR_MAPPINGS[c][1]))

        hiragana_train, hiragana_val = tf.keras.utils.image_dataset_from_directory(
            directory=hiragana_dir,
            labels=None,
            label_mode=None,
            color_mode="grayscale",
            batch_size=None,
            image_size=image_size,
            interpolation="bilinear",
            shuffle=False,
            validation_split=0.1,
            subset="both"
        )

        katakana_train, katakana_val = tf.keras.utils.image_dataset_from_directory(
            directory=katakana_dir,
            labels=None,
            label_mode=None,
            color_mode="grayscale",
            batch_size=None,
            image_size=image_size,
            interpolation="bilinear",
            shuffle=False,
            validation_split=0.1,
            subset="both"
        )

        label_tensor = [i] * len(glob.glob(os.path.join(hiragana_dir, "*.png")))
        labels = tf.data.Dataset.from_tensor_slices(label_tensor)

        train = tf.data.Dataset.zip((hiragana_train, katakana_train, labels))
        val = tf.data.Dataset.zip((hiragana_val, katakana_val, labels))

        datasets.append((train, val))
        label_mapping[i] = c
        
    # Concatenate all the datasets together.
    train_ds, val_ds = datasets.pop(0)
    while datasets:
        next_train_ds, next_val_ds = datasets.pop(0)
        train_ds = train_ds.concatenate(next_train_ds)
        val_ds = val_ds.concatenate(next_val_ds)
    
    # Shuffle so that characters are in random order
    train_ds = train_ds.shuffle(buffer_size=20000, reshuffle_each_iteration=True, seed=42)
    val_ds = val_ds.shuffle(buffer_size=20000, reshuffle_each_iteration=True, seed=42)
    
    # Batch
    train_ds = train_ds.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Preprocess
    train_ds = preprocessing(train_ds)
    val_ds = preprocessing(val_ds)

    # Performance optimization
    train_ds = train_ds.cache()
    val_ds = val_ds.cache()
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, label_mapping
