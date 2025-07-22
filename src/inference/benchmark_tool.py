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
#
#Code is modified from https://github.com/om-ai-lab/RS5M.git

import logging
import open_clip
import torch
import os
from src.inference.classname_and_prompt import *
from src.inference.datasets import RESISC45, EuroSATRGB, AID, EuroSATMS, METERML, METERML_NAIP
from src.inference.clip_benchmark.metrics.zeroshot_eval import evaluate
from src.inference.utils import get_preprocess
import copy
import numpy as np
import os
from torchvision import transforms
import torch
from src.inference.datasets.forestnet import init_forestnet
from src.inference.datasets.bigearthnet import init_bigearthnet


def zeroshot_get_dataset(dataset_name, root, other_features, templates, transform=None, all_bands=False):
    untransformed_dataset = None

    if templates == "msclip":
        prompt_template = MSCLIP.templates
    elif templates == "clip":
        prompt_template = CLIP.templates
    elif templates == "rs5m":
        prompt_template = RS5M.templates

    if dataset_name == "EuroSAT_RGB":
        EuroSAT_root = os.path.join(root, "EuroSAT_RGB")

        os.makedirs(EuroSAT_root, exist_ok=True)
        dataset = EuroSATRGB(
            root=EuroSAT_root,
            transform=transform
        )
        dataset.classes = dataset.classes
        dataset.templates = prompt_template

    if dataset_name == "EuroSAT_MS":

        EuroSAT_root = os.path.join(root, "EuroSAT_MS")
        os.makedirs(EuroSAT_root, exist_ok=True)
        if all_bands:
            lambda_transforms = [transforms.Lambda(lambda x: np.delete(x, [9], axis=2))]
        else:
            lambda_transforms = [transforms.Lambda(lambda x: np.delete(x, [0, 9, 10], axis=2))]

        dataset = EuroSATMS(
            root=EuroSAT_root,
            transform=transforms.Compose(lambda_transforms + transform.transforms)
        )
        untransformed_dataset = EuroSATMS(
            root=EuroSAT_root,
            transform=transforms.Compose([transforms.Lambda(lambda x: x.astype(np.float32)), transforms.ToTensor(),
                                          transforms.Resize(size=224, antialias=True)])
        )

        dataset.classes = dataset.classes
        dataset.templates = prompt_template

    if dataset_name == "METERML_RGB":
        normalize_transform = next(t for t in transform.transforms if isinstance(t, transforms.Normalize))
        mean, std = normalize_transform.mean, normalize_transform.std
        METERML_root = os.path.join(root, "METERML_RGB")

        os.makedirs(METERML_root, exist_ok=True)
        lambda_transforms = transforms.Compose([
            transforms.Lambda(lambda x: x[:, :, [3, 2, 1]]),
            transforms.Lambda(lambda x: x.astype(np.float32) / 2000),
            transforms.Lambda(lambda x: (x * 255).astype(np.uint8)),

            transforms.ToTensor(),  # for rgb the values are scaled but not for ms
            transforms.Resize(
                size=224,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=mean, std=std)
        ])
        dataset = METERML(
            root=METERML_root,
            transform=lambda_transforms
        )
        dataset.classes = RSMETERML.classes
        dataset.templates = prompt_template

    if dataset_name == "METERML_NAIP":
        METERML_root = os.path.join(root, "METERML_NAIP")

        os.makedirs(METERML_root, exist_ok=True)

        dataset = METERML_NAIP(
            root=METERML_root,
            transform=transform
        )
        dataset.classes = RSMETERML.classes
        dataset.templates = prompt_template

    if dataset_name == "METERML_MS":

        METERML_root = os.path.join(root, "METERML_MS")

        os.makedirs(METERML_root, exist_ok=True)
        if all_bands:
            lambda_transforms = [transforms.Lambda(lambda x: np.delete(x, [], axis=2))]
        else:
            lambda_transforms = [transforms.Lambda(lambda x: np.delete(x, [0, 9], axis=2))]
        dataset = METERML(
            root=METERML_root,
            transform=transforms.Compose(lambda_transforms + transform.transforms)
        )
        dataset.classes = RSMETERML.classes
        dataset.templates = prompt_template

        untransformed_dataset = METERML(
            root=METERML_root,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Resize(size=224, antialias=True)])
        )

    elif dataset_name == "AID_RGB":
        AID_root = os.path.join(root, "AID_RGB")
        os.makedirs(AID_root, exist_ok=True)
        dataset = AID(
            root=AID_root,
            transform=transform
        )
        dataset.classes = dataset.classes
        dataset.templates = prompt_template

    elif dataset_name == "RESISC45_RGB":
        RESISC45_root = os.path.join(root, "RESISC45_RGB")
        os.makedirs(RESISC45_root, exist_ok=True)
        dataset = RESISC45(
            root=RESISC45_root,
            transform=transform
        )
        dataset.classes = dataset.classes
        dataset.templates = prompt_template

    elif dataset_name == "BigEarthNet_RGB":
        normalize_transform = next(t for t in transform.transforms if isinstance(t, transforms.Normalize))
        mean, std = normalize_transform.mean, normalize_transform.std
        # mean = (2183.128, 2338.041, 1925.161)
        # std = (1399.638, 1223.713, 1205.586)
        bigearthnet_root = os.path.join(root, "BigEarthNet")
        dataset = init_bigearthnet(bigearthnet_root, [3, 2, 1], True, 19, mean, std, other_features, rgb=True)
        setattr(dataset, "classes", dataset.class_sets[19])
        setattr(dataset, "templates", prompt_template)

    elif dataset_name == "BigEarthNet_MS":
        normalize_transform = next(t for t in transform.transforms if isinstance(t, transforms.Normalize))
        mean, std = normalize_transform.mean, normalize_transform.std
        bigearthnet_root = os.path.join(root, "BigEarthNet")
        if all_bands:
            bands = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  #
        else:
            bands = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
        dataset = init_bigearthnet(bigearthnet_root, bands, True, 19, mean, std, other_features)
        untransformed_dataset = init_bigearthnet(bigearthnet_root, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], False, 19,
                                                 mean, std, other_features)
        setattr(dataset, "classes", dataset.class_sets[19])
        setattr(dataset, "templates", prompt_template)

    elif dataset_name == "ForestNet_RGB":
        normalize_transform = next(t for t in transform.transforms if isinstance(t, transforms.Normalize))
        mean, std = normalize_transform.mean, normalize_transform.std
        forestnet_root = os.path.join(root, "ForestNet_RGB")
        dataset = init_forestnet(forestnet_root, True, mean, std)
        setattr(dataset, "classes", dataset.label_names)
        setattr(dataset, "templates", prompt_template)

    if dataset_name not in ["BigEarthNet_RGB", "ForestNet_RGB", "BigEarthNet_MS"]:
        dataset.classes = [dataset.classes[i].replace('_', ' ') for i in range(len(dataset.classes))]
        dataset.classes = [dataset.classes[i].replace('/', ' ') for i in range(len(dataset.classes))]
        dataset.classes = [dataset.classes[i].lower() for i in range(len(dataset.classes))]

    return dataset, untransformed_dataset


def zeroshot_evaluation(model, zeroshot_dataset, preprocess, args):
    all_bands = hasattr(model, "channels") and model.channels == 12
    dataset, _ = zeroshot_get_dataset(dataset_name=zeroshot_dataset, root=args.dataset_dir,
                                      other_features=args.other_features, templates=args.templates,
                                      transform=preprocess, all_bands=all_bands)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
    logging.info(f'Calculating classifier for {zeroshot_dataset}')
    classnames, prompt_templates = dataset.classes, dataset.templates
    one_class = not zeroshot_dataset in ["BigEarthNet", "BigEarthNetMS"]

    if "Llama3-MS-CLIP" in args.model_name:
        tokenizer = open_clip.get_tokenizer('ViT-B-16')
    else:
        tokenizer = open_clip.get_tokenizer(args.model_name)
    classnames = copy.deepcopy(classnames)

    clip_benchmark_metrics = evaluate(model, dataloader, tokenizer, classnames, prompt_templates, args.device,
                                      one_class=one_class, other_features=args.other_features)
    print(clip_benchmark_metrics)

    return clip_benchmark_metrics
