"""
Entry point
"""

import argparse
from datetime import datetime
import os

import yaml

from dataset import create_dataset
from sample_dataset import sample_dataset
from training.train_classifiers import train as train_classifiers
from training.train_generators import train as train_generators
from predict import predict
from visualize_capsules import visualize_capsules

if __name__ == "__main__":
    args = argparse.Namespace()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["sample_dataset", "train", "predict", "visualize_capsules"]
    )
    parser.parse_known_args(namespace=args)

    with open("./config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    timestamp = datetime.now().strftime(r"%d%m%Y%H%M%S")

    if args.mode == "sample_dataset":
        sample_dataset(config)

    elif args.mode == "train":
        train_dataset, val_dataset, label_mapping = create_dataset(config)

        hiragana_classifier, katakana_classifier = train_classifiers(
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            label_mapping=label_mapping,
            timestamp=timestamp
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
            val_dataset=val_dataset,
            timestamp=timestamp
        )

    elif args.mode == "predict":
        predict_parser = argparse.ArgumentParser()
        predict_parser.add_argument(
            "--model_version",
            type=str,
            required=True,
            help="Model version to use"
        )
        predict_parser.add_argument(
            "--filepath",
            type=str,
            required=True,
            help="Location of input image to run prediction/generation"
        )
        predict_parser.add_argument(
            "--kana_type",
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
        predict_parser.add_argument(
            "--loops",
            type=int,
            default=1,
            help="Number of times to apply the generator"
        )
        predict_parser.parse_known_args(namespace=args)

        # Load the config from the model.
        with open(
            os.path.join(config["config_save_path"], args.model_version, "config.yaml"),
            "r"
        ) as stream:
            config = yaml.safe_load(stream)

        predict(
            config,
            model_version=args.model_version,
            filepath=args.filepath,
            kana_type=args.kana_type,
            reps=args.reps,
            loops=args.loops
        )
    
    else:
        vis_parser = argparse.ArgumentParser()
        vis_parser.add_argument(
            "--model_version",
            type=str,
            required=True,
            help="Model version to use"
        )
        vis_parser.add_argument(
            "--filepath",
            type=str,
            required=True,
            help="Location of input image to run visualization"
        )
        vis_parser.add_argument(
            "--kana_type",
            choices=["h", "k"],
            required=True,
            help="Type of input character image. h for hiragana, k for katakana"
        )

        visualize_capsules(
            config,
            model_version=args.model_version,
            filepath=args.filepath,
            kana_type=args.kana_type,
        )