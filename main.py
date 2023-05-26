"""
Entry point
"""

import argparse

import yaml

from dataset import create_dataset
from sample_dataset import sample_dataset
from training.train_classifiers import train as train_classifiers
from training.train_generators import train as train_generators
from predict import predict

if __name__ == "__main__":
    args = argparse.Namespace()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["sample_dataset", "train", "predict"]
    )
    parser.parse_known_args(namespace=args)

    with open("./config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    if args.mode == "sample_dataset":
        sample_dataset(config)

    elif args.mode == "train":
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
        predict_parser = argparse.ArgumentParser()
        predict_parser.add_argument(
            "--filepath",
            type=str,
            required=True,
            help="Location of input image to run prediction/generation"
        )
        predict_parser.add_argument(
            "--type",
            choices=["h", "k"],
            required=True,
            help="Type of input character image. h for hiragana, k for katakana"
        )
        predict_parser.add_argument(
            "--reps",
            type=int,
            default=1,
            help="Number of times to apply the generator"
        )
        predict_parser.parse_known_args(namespace=args)

        predict(config, filepath=args.filepath, type=args.type, reps=args.reps)
