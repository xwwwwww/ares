#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import pytorch_lightning as pl
import json
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), "example")
MODELS = [
    "cifar10-pgd_at",
    "cifar10-wideresnet_trades",
    "cifar10-feature_scatter",
    "cifar10-robust_overfitting",
    "cifar10-rst",
    "cifar10-fast_at",
    "cifar10-at_he",
    "cifar10-pre_training",
    "cifar10-mmc",
    "cifar10-free_at",
    "cifar10-awp",
    "cifar10-hydra",
    "cifar10-label_smoothing",
    "imagenet-fast_at",
    "imagenet-free_at",
]
BATCH_SIZE = 50


def main(args):
    attack_names = [x for x in args.attacks.split(",") if len(x) > 0]
    model_names = [x for x in args.models.split(",") if len(x) > 0]
    output_directory = os.path.abspath(args.output)
    print("Models: ", model_names)
    print("Attacks: ", attack_names)
    print("Output Directory: ", output_directory)
    for model_name in model_names:
        p = subprocess.Popen(
            [
                sys.executable, "run_attacks_loader.py",
                "--model", model_name,
                "--attacks", ",".join(attack_names),
                "--output", output_directory,
            ],
            stdout=sys.stdout,
            stderr=subprocess.DEVNULL if args.mute_stderr else subprocess.STDOUT,
        )
        code = p.wait()
        if code != 0:
            sys.exit(code)

    final_eval(os.path.join(output_directory, args.name))


def final_eval(path):
    files = os.listdir(path)
    result = {}
    for file in files:
        if file.endswith(".json"):
            continue
        file_path = os.path.join(path, file)
        with open(file_path, 'r') as f:
            score = f.readline().strip()
        result[file.split('.')[0]] = float(score)
        # os.remove(file_path)
    result['avg'] = np.mean(list(result.values())).item()
    with open(os.path.join(path, 'result.json'), 'w') as f:
        json.dump(result, f, indent=4, separators=[',', ':'])
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        help="comma-separated list of models to run in the format of '<dataset>-<model>', e.g. 'imagenet-free_at', "
             "default to all models used by competition stage I, they are {}".format(", ".join(MODELS)),
        default=",".join(MODELS),
    )
    parser.add_argument(
        "--attacks",
        help="comma-separated list of attack folder/package names to run, default to 'attacker'",
        default="attacker",
    )
    parser.add_argument(
        "--name",
        help="name of attacker, such as pgd, bim",
        required=True,
        default="pgd",
    )
    parser.add_argument(
        "--output",
        help="output directory, default to the current directory",
        default="tmp",
    )
    parser.add_argument(
        "--mute-stderr",
        help="mute stderr",
        action="store_true",
        default=False,
    )
    pl.seed_everything(20210523)
    args = parser.parse_args()
    main(args)
