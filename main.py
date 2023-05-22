"""
Entry point
"""

import argparse

import yaml

from .dataset import create_dataset
from .training.train_classifiers import train as train_classifiers
from .training.train_generators import train as train_generators

if __name__ == "__main__":
    args = argparse.Namespace()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "generate"]
    )
    parser.parse_known_args(namespace=args)

    with open("./config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    if args.mode == "train":
        train_dataset, val_dataset, label_mapping = create_dataset(config)

        hiragana_classifier, katakana_classifier = train_classifiers(
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            label_mapping=label_mapping
        )

        (
            hiragana_classifier,
            katakana_classifier,
            hiragana_generator,
            katakana_generator
        ) = train_generators(
            config,
            hiragana_classifier=hiragana_classifier,
            katakana_classifier=katakana_classifier,
            train_dataset=train_dataset,
            val_dataset=val_dataset
        )

    else:
        file_parser = argparse.ArgumentParser()
        file_parser.add_argument(
            "--file",
            type=str,
            required=True,
            help="Location of input image to run prediction/generation"
        )
        file_parser.add_argument(
            "--type",
            choices=["h", "k"],
            required=True,
            help="Type of input character image. h for hiragana, k for katakana"
        )
        file_parser.parse_known_args(namespace=args)

        raise NotImplemented("Prediction and Generation not yet implemented ;)")
    