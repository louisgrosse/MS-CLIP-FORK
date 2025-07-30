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

import os
import argparse
import warnings

import torch
import logging
import open_clip
import pandas as pd
from pathlib import Path
from tabulate import tabulate
from msclip.inference.utils import open_clip_weights, pretrained_weights

from msclip.inference.utils import (
    build_model,
    preprocess_and_stack,
    load_classes,
    load_templates,
    load_image_paths,
    load_queries
)
from msclip.inference.clip_benchmark.metrics.zeroshot_eval import zero_shot_classifier

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


def run_inference_classification(model, classifier, image_paths, device, preprocess, class_names=None):
    results = []
    if isinstance(image_paths, str) and os.path.isdir(image_paths):
        image_paths = [
            os.path.join(image_paths, f) for f in os.listdir(image_paths)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".npy", ".npz"))
        ]

    image_tensor = preprocess_and_stack(image_paths, preprocess)  # [B, C, H, W]
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        image_features = model.inference_vision(image_tensor)  # [B, D]
        image_features = torch.nn.functional.normalize(image_features, dim=-1)

        logits = 100. * image_features @ classifier  # [B, num_classes]
        probs = logits.softmax(dim=-1).cpu().numpy()  # [B, num_classes]

    for path, prob in zip(image_paths, probs):
        results.append((os.path.basename(path), prob))

    # Format nicely into a DataFrame
    if class_names is None:
        class_names = [f"Class {i}" for i in range(probs.shape[1])]

    df = pd.DataFrame([dict(zip(class_names, prob)) for _, prob in results])
    df.insert(0, "Class", df.idxmax(axis=1))
    df.insert(0, "Image", [name for name, _ in results])

    print(f"Zero-Shot Classification Results:")
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', floatfmt=".3f"))

    return results


def run_inference_retrieval(model, text_queries, image_paths, device, preprocess, tokenizer, top_k=1):
    # Encode image features

    if isinstance(image_paths, str) and os.path.isdir(image_paths):
        image_paths = [
            os.path.join(image_paths, f) for f in os.listdir(image_paths)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".npy", ".npz"))
        ]
    image_tensor = preprocess_and_stack(image_paths, preprocess)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        image_features = model.inference_vision(image_tensor)  # [B, D]
        image_features = torch.nn.functional.normalize(image_features, dim=-1)

        # Encode text
        text_tokens = tokenizer(text_queries).to(device)
        text_features = model.inference_text(text_tokens)  # [T, D]
        text_features = torch.nn.functional.normalize(text_features, dim=-1)

        # Similarity: [num_images, num_queries]
        similarity = image_features @ text_features.T  # [B, T]

    retrieval_results = pd.DataFrame(columns=["Query", "Rank", "Image", "Similarity"])
    for i, query in enumerate(text_queries):
        topk = torch.topk(similarity[:, i], k=top_k)
        topk_indices = topk.indices.tolist()
        topk_scores = topk.values.tolist()

        topk_items = [
            (query, rank+1, os.path.basename(image_paths[idx]), score)
            for rank, (idx, score) in enumerate(zip(topk_indices, topk_scores))
        ]

        topk_items = pd.DataFrame(topk_items, columns=["Query", "Rank", "Image", "Similarity"])
        retrieval_results = pd.concat([retrieval_results, topk_items])

    retrieval_results = retrieval_results.set_index(["Query", "Rank"])

    print(f"Text-to-Image Retrieval (Top-{top_k}) Results:")
    print(retrieval_results)

    return retrieval_results


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot CLIP inference")

    parser.add_argument("--model-name", default="Llama3-MS-CLIP-Base", type=str,
        help="Our model 'Llama3-MS-CLIP-Base' (default), OpenCLIP model names: 'ViT-B-16', 'ViT-L-14', or 'ViT-B-32'")
    parser.add_argument("--pretrained", default=True, type=bool, help="Load pretrained model")
    parser.add_argument("--ckpt-path", default=None, type=str, help="Path to ckpt.pt file",)
    parser.add_argument('--images', type=str, default="./examples",
                        help='Path to .yaml or .txt or simple directory string with image paths')
    parser.add_argument('--save-path', type=str, help="Optinal path to file for saving inference results")

    # Classification args
    parser.add_argument('--run-classification', action="store_true",
                        help='Run classification for inferencing')
    parser.add_argument('--class-names', type=str, nargs='*', help='List your class names')
    parser.add_argument('--classes-file', type=str, help='Path to class names .yaml or .txt file')
    parser.add_argument('--templates', type=str, default="msclip", help='Prompt template to use')

    # Retrieval args
    parser.add_argument('--run-retrieval', action="store_true",
                        help='Run image to text retrieval for inferencing')
    parser.add_argument('--query', type=str, help='Query text for retrieval task')
    parser.add_argument('--queries-file', type=str, help='Path to text queries .yaml or .txt file')
    parser.add_argument('--top-k', type=int, default=5, help='Number of top images to retrieve per query')

    parser.add_argument('--amp', default=True, help='Use automatic mixed precision')

    return parser.parse_args()


def main():
    args = parse_args()

    model, img_preprocess = build_model(args.model_name, args.pretrained, args.ckpt_path, device)

    # Get base model from the model architecture
    if args.model_name in open_clip_weights:
        base_model = args.model_name
    elif args.model_name in pretrained_weights:
        base_model = pretrained_weights[args.model_name]["architecture"]
    else:
        raise ValueError("Cannot find base model architecture from model name.")

    tokenizer = open_clip.get_tokenizer(base_model)

    image_paths = load_image_paths(args.images)

    if args.run_classification:
        # Class and prompt templates
        templates = load_templates(args.templates)
        if len(args.class_names):
            classnames = args.class_names
        elif args.classes_file is not None:
            load_classes(args.classes_file)
        else:
            raise ValueError("No class names provided, please use --class_names or --classes_file")

        classifier = zero_shot_classifier(
            model, tokenizer, classnames, templates, device=device, amp=args.amp
        )

        result = run_inference_classification(model, classifier, image_paths, device, img_preprocess, classnames)

        if args.save_path:
            save_path = Path(args.save_path).with_suffix(".csv")
            save_path.parent.mkdir(exist_ok=True, parents=True)
            result.to_csv(save_path)
            print(f"Saved zero-shot classification results to {save_path}")

            assert not args.run_retrieval, \
                "Please run classification and retrieval separate if you are providing a save_path."


    if args.run_retrieval:
        if args.query:
            text_queries = [args.query]
        elif args.queries_file is not None:
            text_queries = load_queries(args.queries_file)
        else:
            raise ValueError("No queries provided, please use --query or --queries_file")

        result = run_inference_retrieval(model, text_queries, image_paths, device, img_preprocess, tokenizer,
                                         top_k=args.top_k)
        if args.save_path:
            save_path = Path(args.save_path).with_suffix(".csv")
            save_path.parent.mkdir(exist_ok=True, parents=True)
            result.to_csv(save_path)
            print(f"Saved text-to-image retrieval results to {save_path}")


if __name__ == "__main__":
    main()
