"""
Dataset processing functions
"""
import glob
import itertools
import os
import random
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

    return hiragana_images, katakana_images


@tf.function
def remove_noise(img, threshold, mode="mean"):
    """
    Remove image noise by applying mean filter
    `img` is a Tensor of shape [batch, height, width, channels]
    Returns a Tensor of same shape.
    """
    img_shape = tf.shape(img)

    # hardcoded threshold
    new_img = 255 * tf.cast(img > threshold, dtype=img.dtype)

    patches = tf.image.extract_patches(
        new_img,
        sizes=[1, 3, 3, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding="SAME"
    )
    patches = tf.reshape(patches, tf.concat([img_shape, [-1]], axis=0))

    if mode == "mean":
        kernel = tf.constant([0.075, 0.075, 0.075, 0.075, 0.4, 0.075, 0.075, 0.075, 0.075], dtype=patches.dtype)
        patches = tf.reduce_sum(patches*kernel[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :], axis=-1, keepdims=True)
    else: # "median"
        patches = tf.sort(patches, axis=-1)
        median_idx = tf.math.ceil(img_shape[-1] / 2)
        patches = tf.slice(
            patches,
            begin=tf.cast(tf.concat([[0,0,0,0], median_idx[tf.newaxis]], axis=0), tf.int32),
            size=tf.concat([img_shape, [1]], axis=0)
        )

    return tf.squeeze(patches, axis=-1)


@tf.function
def apply_crop(img, hiragana=False):
    """
    Crop to bounding box.
    Hiragana (ETLCDB-4) appears to have different bounding box
    than Katakana (ETLCDB-5).

    The hiragana box is smaller - we resize the cropped katakana images
    to match the hiragana box size.
    """
    if hiragana:
        return tf.image.crop_to_bounding_box(
            img,
            offset_height=9,
            offset_width=17,
            target_height=52,
            target_width=36,
        )

    cropped = tf.image.crop_to_bounding_box(
        img,
        offset_height=1,
        offset_width=15,
        target_height=65,
        target_width=46,
    )

    return tf.image.resize(
        cropped,
        size=(52, 36)
    )


def preprocessing(dataset, crop_image=False, predict=None):
    """
    Apply preprocessing transformations to the dataset.
    """

    # Crop to bounding box
    if crop_image:
        dataset = dataset.map(
            lambda *t:
                (
                    apply_crop(t[0], hiragana=True),
                    apply_crop(t[1], hiragana=False),
                    t[2]
                ) if predict is None
                else apply_crop(t[0], hiragana=(predict=="h")),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    # Invert - white characters on black background
    dataset = dataset.map(
        lambda *t: 
            (-1*t[0]+255,-1*t[1]+255, t[2]) if predict is None
            else -1*t[0]+255,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Remove noise
    dataset = dataset.map(
        lambda *t:
            (
                # different datasets, different thresholds
                remove_noise(t[0], 95),
                remove_noise(t[1], 70),
                t[2]
            ) if predict is None
            else remove_noise(t[0], 95 if predict=="h" else 70),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # scale to 0 - 1
    dataset = dataset.map(
        lambda *t:
            (t[0]/255, t[1]/255, t[2]) if predict is None
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
            `hiragana_image`: Tensor of shape [batch, height, width, channels]
            `katakana_image`: Tensor of shape [batch, height, width, channels]
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
    train_ds = preprocessing(train_ds, crop_image=config["crop_image"])
    val_ds = preprocessing(val_ds, crop_image=config["crop_image"])

    # Performance optimization
    train_ds = train_ds.cache()
    val_ds = val_ds.cache()
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, label_mapping


def load_img_to_array(filepath):
    """
    Loads image from filepath
    """
    img = tf.image.decode_png(
        tf.io.read_file(filepath),
        channels=1
    )
    
    return tf.cast(img, tf.float32)

def create_dataset_v2(
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
            `hiragana_image`: Tensor of shape [batch, height, width, channels]
            `katakana_image`: Tensor of shape [batch, height, width, channels]
            `labels`: Tensor of shape [batch, 1] containing integer labels

        label mapping: Dictionary mapping integer label to string character.
    """

    data_dir = config["data_dir"]
    characters = config["characters"]
    batch_size = config["batch_size"]

    label_mapping = {}
    all_train_pairs = []
    all_val_pairs = []
    for i, c in enumerate(characters):
        hiragana_files, katakana_files = validate_subdirectories(data_dir, c)
        random.shuffle(hiragana_files)
        random.shuffle(katakana_files)

        train_num = int(len(hiragana_files)*.9)
        hiragana_files_train = hiragana_files[:train_num]
        hiragana_files_val = hiragana_files[train_num:]
        katakana_files_train = katakana_files[:train_num]
        katakana_files_val = katakana_files[train_num:]
        
        train_pairs = itertools.product(hiragana_files_train, katakana_files_train)
        val_pairs = itertools.product(hiragana_files_val, katakana_files_val)
        label_iter = itertools.repeat(i)
        
        # shuffle - not memory efficient but whatever, dataset is small
        train_pairs = list(train_pairs)
        val_pairs = list(val_pairs)
        random.shuffle(train_pairs)
        random.shuffle(val_pairs)

        train_pairs = ((h, k, l) for (h, k), l in zip(train_pairs, label_iter))
        val_pairs = ((h, k, l) for (h, k), l in zip(val_pairs, label_iter))

        all_train_pairs.append(train_pairs)
        all_val_pairs.append(val_pairs)
        label_mapping[i] = c

    all_train_pairs = itertools.chain(*all_train_pairs)
    all_val_pairs = itertools.chain(*all_val_pairs)

    def train_gen():
        for p in all_train_pairs:
            yield p

    def val_gen():
        for p in all_val_pairs:
            yield p

    # Load the images
    train_ds = tf.data.Dataset.from_generator(
        generator=train_gen,
        output_types = (tf.string, tf.string, tf.int32)
    )
    val_ds = tf.data.Dataset.from_generator(
        generator=val_gen,
        output_types = (tf.string, tf.string, tf.int32)
    )

    train_ds = train_ds.map(
        lambda h, k, l: (
            load_img_to_array(h),
            load_img_to_array(k),
            l
        )
    )
    val_ds = val_ds.map(
        lambda h, k, l: (
            load_img_to_array(h),
            load_img_to_array(k),
            l
        )
    )

    train_ds = train_ds.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = preprocessing(train_ds, crop_image=config["crop_image"])
    val_ds = preprocessing(val_ds, crop_image=config["crop_image"])

    # Performance optimization
    train_ds = train_ds.cache()
    val_ds = val_ds.cache()
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, label_mapping
