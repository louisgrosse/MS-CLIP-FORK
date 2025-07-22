# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import torch
import os
import random
import numpy as np
import argparse
from pathlib import Path
from src.inference.benchmark_tool import zeroshot_evaluation
from src.inference.utils import build_model
import csv
from datetime import datetime

logging.basicConfig(
    level=os.getenv('log_level', 'INFO'),
    handlers=[logging.StreamHandler()],
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
else:
    device = "cpu"


def random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def write_to_csv(metrics_dict, dataset, args):
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_file = f"{args.save_name}_{current_time}" or f"{args.model_name}_{current_time}_results"
    csv_file = (Path(args.save_path) / save_file).with_suffix('.csv')
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    # Extract keys and values as rows
    rows = [(key, value) for key, value in metrics_dict.items()]

    # Write to CSV
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

    print(f'Dictionary data has been written to {csv_file}.')


def evaluate_rgb(model, preprocess, args):
    print("making val dataset with transformation for rgb: ")
    print(preprocess)
    zeroshot_datasets = [
        "METERML_RGB",
        "EuroSAT_RGB",
        "ForestNet_RGB",
        "RESISC45_RGB",
        "AID_RGB",
        "BigEarthNet_RGB",
        "METERML_NAIP",
    ]

    model.eval()
    all_metrics = {}

    metrics_cl = {}
    for zeroshot_dataset in zeroshot_datasets:
        try:
            zeroshot_metrics = zeroshot_evaluation(model, zeroshot_dataset, preprocess, args)
        except Exception as e:
            logging.info(f'Skipping {zeroshot_dataset}, evaluation failed with error {type(e).__name__}: {e}')
            continue

        write_to_csv(zeroshot_metrics, zeroshot_dataset, args)
        metrics_cl.update(zeroshot_metrics)
        all_metrics.update(zeroshot_metrics)

    return all_metrics


def evaluate_ms(model, preprocess, args):
    print("making val dataset with transformation for ms datasets : ")
    print(preprocess)
    zeroshot_datasets = [
        "METERML_MS",
        "EuroSAT_MS",
        "BigEarthNet_MS"
    ]

    model.eval()
    all_metrics = {}

    metrics_cl = {}
    for zeroshot_dataset in zeroshot_datasets:
        zeroshot_metrics = zeroshot_evaluation(model, zeroshot_dataset, preprocess, args)
        write_to_csv(zeroshot_metrics, zeroshot_dataset, args)
        metrics_cl.update(zeroshot_metrics)
        all_metrics.update(zeroshot_metrics)

    return all_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name", default="Llama3-MS-CLIP-Base", type=str,
        help="Our model 'Llama3-MS-CLIP-Base' (default), OpenCLIP model names: 'ViT-B-16', 'ViT-L-14', or 'ViT-B-32'",
    )
    parser.add_argument(
        "--pretrained", default=True, type=bool,
        help="Load pretrained model",
    )
    parser.add_argument(
        "--ckpt-path", default=None, type=str,
        help="Path to ckpt.pt file",
    )
    parser.add_argument(
        "--save-name", type=str,
        help="Unique saving name for results csv. Using model name by default.",
    )
    parser.add_argument(
        "--other-features", action="store_true",
        help="Whether to use extra class in BEN called other physical features for classification",
    )
    parser.add_argument(
        "--templates", default="msclip", type=str,
        help="prompting templates to use",
    )
    parser.add_argument(
        "--random-seed", default=3407, type=int,
        help="random seed",
    )
    parser.add_argument(
        "--dataset-dir", default="benchmark_datasets", type=str,
        help="location of benchmark datasets",
    )
    parser.add_argument(
        "--batch-size", default=64, type=int,
        help="batch size",
    )
    parser.add_argument(
        "--workers", default=0, type=int,
        help="number of workers")
    parser.add_argument(
        "--precision", default="amp", type=str)
    parser.add_argument(
        "--save-path", type=str, default="results/",
        help="Directory for saving checkpoints and results"
    )
    parser.add_argument(
        "--device", type=str, default=device, choices=["cuda", "mps", "cpu"],
        help=f"Device to use (default: {device})"
    )

    args = parser.parse_args()

    model, img_preprocess = build_model(args.model_name, args.pretrained, args.ckpt_path, args.device)
    if "Llama3-MS-CLIP" in args.model_name:
        # Multi-spectral evaluation
        eval_result = evaluate_ms(model, img_preprocess, args)
    else:
        eval_result = evaluate_rgb(model, img_preprocess, args)


if __name__ == "__main__":
    main()
