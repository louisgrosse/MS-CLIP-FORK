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

import argparse
from msclip.inference import run_inference_classification, run_inference_retrieval


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot CLIP inference")

    parser.add_argument("--model-name", default="Llama3-MS-CLIP-Base", type=str,
                        help="Our model 'Llama3-MS-CLIP-Base' (default), OpenCLIP model names: 'ViT-B-16', 'ViT-L-14', or 'ViT-B-32'")
    parser.add_argument("--pretrained", default=True, type=bool, help="Load pretrained model")
    parser.add_argument("--ckpt-path", default=None, type=str, help="Path to ckpt.pt file", )
    parser.add_argument("--images", type=str, default="./examples",
                        help="Path to .yaml or .txt or simple directory string with image paths")
    parser.add_argument("--save-path", type=str, help="Optinal path to file for saving inference results")

    # Classification args
    parser.add_argument("--run-classification", action="store_true",
                        help="Run classification for inferencing")
    parser.add_argument("--class-names", type=str, nargs="*", help="List your class names")
    parser.add_argument("--classes-file", type=str, help="Path to class names .yaml or .txt file")
    parser.add_argument("--templates", type=str, default="msclip", help="Prompt template to use")

    # Retrieval args
    parser.add_argument("--run-retrieval", action="store_true",
                        help="Run image to text retrieval for inferencing")
    parser.add_argument("--query", type=str, nargs="*", help="Query text for retrieval task")
    parser.add_argument("--queries-file", type=str, help="Path to text queries .yaml or .txt file")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top images to retrieve per query")

    parser.add_argument("--amp", default=True, help="Use automatic mixed precision")
    parser.add_argument("--device", default=None, help="Specify device")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.run_classification:
        _ = run_inference_classification(
            model_name=args.model_name,
            pretrained=args.pretrained,
            ckpt_path=args.ckpt_path,
            image_path=args.images,
            class_names=args.class_names,
            classes_file=args.classes_file,
            save_path=args.save_path,
            device=args.device,
            amp=args.amp,
            templates=args.templates,
            verbose=True,
        )

        assert not args.save_path or not args.run_retrieval, \
            "Please run classification and retrieval separate if you are providing a save_path."

    if args.run_retrieval:
        _ = run_inference_retrieval(
            model_name=args.model_name,
            pretrained=args.pretrained,
            ckpt_path=args.ckpt_path,
            image_path=args.images,
            queries=args.query,
            queries_file=args.queries_file,
            top_k=args.top_k,
            save_path=args.save_path,
            device=args.device,
            amp=args.amp,
            verbose=True,
        )


if __name__ == "__main__":
    main()
