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
import warnings

import os
import torch
import open_clip
import logging
import yaml
import tifffile
import numpy as np
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download
from src.model.model_arch import CLIPDualEncoderModel
from src.inference.classname_and_prompt import *


open_clip_weights = {
    "ViT-B-16": "laion2b-s34b-b88K",
    "ViT-L-14": "laion2B-s32B-b82K",
    "ViT-B-32": "laion2B-s34B-b79K",
}

pretrained_weights = {
    "Llama3-MS-CLIP-Base": {
        "architecture": "ViT-B-16",
        "hf_hub_id": "ibm-esa-geospatial/Llama3-MS-CLIP-base",
        "hf_hub_filename": "Llama3_MS_CLIP_weights.pt",
    },
    "Llama3-RGB-CLIP-Base": {
        "architecture": "ViT-B-16",
    }
}

pretrained_cfg = {
    "Llama3-MS-CLIP-Base": {
        "base_model_str": "ViT-B-16",
        "ckpt": "laion2b_s34b_b88K",
        "channels": 10,
    }
}


def build_model(model_name, pretrained, ckpt_path, device, **kwargs):
    if model_name in pretrained_weights:
        if ckpt_path:
            # Local model
            pretrained = ckpt_path
        elif pretrained:
            # Load Llama3-MS-CLIP from Hugging Face
            pretrained = hf_hub_download(repo_id=pretrained_weights[model_name]['hf_hub_id'],
                                         filename=pretrained_weights[model_name]['hf_hub_filename'])
        logging.info(f"Initializing {model_name} model (checkpoint: {os.path.basename(pretrained)})")

        cfg = pretrained_cfg[model_name]
        if pretrained:
            cfg["ckpt"] = None  # Avoid loading weights twice
        cfg.update(kwargs)  # Update config based on kw args

        # Init CLIP model
        model = CLIPDualEncoderModel(**cfg)

        # Load pre-trained weights
        if pretrained:
            model.load_state_dict(torch.load(pretrained, map_location=device), strict=True)

        preprocess_val = get_preprocess(
            image_resolution=224, is_ms=cfg["channels"] > 3, all_bands=cfg["channels"] == 12,
        )

    elif model_name in open_clip_weights:
        if ckpt_path:
            # Custom RS models
            pretrained = ckpt_path
        elif pretrained:
            # Load OpenCLIP model
            pretrained = open_clip_weights[model_name]
        else:
            pretrained = None

        logging.info(f"Initializing {model_name} model (checkpoint: {pretrained})")
        model, _, preprocess_val = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained
        )

    else:
        raise ValueError(f"model_name {model_name} not found in pretrained weights "
        f"(f{list(pretrained_weights.keys()) + list(open_clip_weights.keys())})")

    model.eval()
    model = model.to(device)

    return model, preprocess_val


def load_queries(class_file):
    if class_file.endswith(".txt"):
        with open(class_file, "r") as f:
            return [line.strip() for line in f if line.strip()]
    elif class_file.endswith((".yaml", ".yml")):
        with open(class_file, "r") as f:
            return yaml.safe_load(f)["queries"]
    else:
        raise ValueError("Unsupported class file format. Use .txt or .yaml")


def load_classes(class_file):
    if class_file.endswith(".txt"):
        with open(class_file, "r") as f:
            return [line.strip() for line in f if line.strip()]
    elif class_file.endswith((".yaml", ".yml")):
        with open(class_file, "r") as f:
            return yaml.safe_load(f)["classes"]
    else:
        raise ValueError("Unsupported class file format. Use .txt or .yaml")


def load_image_paths(class_file):
    if class_file.endswith(".txt"):
        with open(class_file, "r") as f:
            return [line.strip() for line in f if line.strip()]
    elif class_file.endswith((".yaml", ".yml")):
        with open(class_file, "r") as f:
            return yaml.safe_load(f)["images"]
    elif os.path.isdir(class_file):
        return class_file  # Just return the directory path string
    else:
        raise ValueError("Unsupported class file format. Use .txt or .yaml")


def load_templates(templates):
    if templates == "msclip":
        prompt_template = MSCLIP.templates
    elif templates == "clip":
        prompt_template = CLIP.templates
    elif templates == "rs5m":
        prompt_template = RS5M.templates
    return prompt_template


def load_image_file(path):
    ext = os.path.splitext(path)[-1].lower()
    if ext in ['.tif', '.tiff']:
        return load_tiff_image(path)
    elif ext == '.npy':
        return load_npy_image(path)
    elif ext == '.npz':
        return load_npz_image(path)
    elif ext in ['.png', '.jpg', '.jpeg']:
        return load_jpg_image(path)
    else:
        raise ValueError(f"Unsupported image format: {ext}")


def load_tiff_image(path):
    image = tifffile.imread(path)
    if image.ndim == 2:
        image = image[None, ...]
    elif image.ndim == 3 and image.shape[-1] <= 10:
        image = np.moveaxis(image, -1, 0)
    return image


def load_npy_image(path):
    image = np.load(path)
    if image.ndim == 2:
        image = image[None, ...]
    return image


def load_npz_image(path):
    npz = np.load(path)
    key = list(npz.keys())[0]
    return npz[key]


def load_jpg_image(path):
    image = Image.open(path)  # Ensure 3-channel RGB
    return image


def preprocess_and_stack(paths, preprocess):
    # Handle folder input
    batch = []

    for path in paths:
        image = load_image_file(path)
        if isinstance(image, np.ndarray):  #only multispectral is numpy array
            if image.shape[2] == 13:
                image = np.delete(image, [0, 9, 10], axis=2)
            elif image.shape[2] == 12:
                image = np.delete(image, [0, 9], axis=2)  # shape: [H, W, C]
        tensor = preprocess(image)  # assume preprocess outputs [C, H, W]
        batch.append(tensor)
    return torch.stack(batch)


def _convert_to_rgb(image):
    return image.convert('RGB')


def get_preprocess(image_resolution=224, is_ms=False, aug=None, all_bands=False):
    with open('configs/data_config.yaml', 'r') as file:
        data_params = yaml.safe_load(file)

        if is_ms:
            if all_bands:
                data_params = data_params["ms_all"]
            else:
                data_params = data_params["ms"]
            preprocess_ms = transforms.Compose([
                transforms.Lambda(lambda x: x.astype(np.float32)),

                transforms.ToTensor(),  #for rgb the values are scaled but not for ms
                transforms.Resize(
                    size=data_params["size"],
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(data_params["size"]),
                transforms.Normalize(mean=[value - 1000 for value in data_params["means"]], std=data_params["stds"]),
                #put back 1000
            ])

            return preprocess_ms

        else:
            data_params = data_params["rgb"]
            normalize = transforms.Normalize(
                mean=data_params["means"], std=data_params["stds"]
            )
            preprocess_rgb = transforms.Compose([
                transforms.Resize(
                    size=data_params["size"],
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(data_params["size"]),
                _convert_to_rgb,
                transforms.ToTensor(),
                normalize,
            ])
            return preprocess_rgb


class DictTransforms:
    def __init__(self,
                 dict_transform: dict,
                 ):
        self.dict_transform = dict_transform

    def __call__(self, sample):
        # Apply your transforms to the 'image' key
        for key, function in self.dict_transform.items():
            sample[key] = function(sample[key])
        return sample


class SelectChannels:
    def __init__(self, channels):
        self.channels = channels

    def __call__(self, tensor):
        return tensor[self.channels]


class Unsqueeze:
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, tensor):
        return tensor.unsqueeze(dim=self.dim)


class ConvertType:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, tensor):
        return tensor.to(self.dtype)


class AddMeanChannels:
    """
    Add missing channels to the tensor based on the mean values. Results in zeros after standardization.
    """

    def __init__(self, mean, fill):
        self.mean = mean
        self.mean_tensor = None
        self.zero_tensor = None
        self.fill = fill

    def __call__(self, tensor):
        if self.fill == 'channel_mean' or self.fill == 'channel_drop':
            if self.mean_tensor is None:
                # Init tensor with mean values
                self.mean_tensor = (torch.ones([len(self.mean) - len(tensor), *tensor.shape[1:]]) *
                                    torch.tensor(self.mean)[len(tensor):, None, None])
            # Add mean values for missing channels
            tensor = torch.concat([tensor, self.mean_tensor])
        elif self.fill == 'pixel_mean':
            fill_tensor = tensor.mean(axis=0, keepdim=True).repeat(len(self.mean) - len(tensor), 1, 1)
            tensor = torch.concat([tensor, fill_tensor])
        elif self.fill == 'zero':
            if self.zero_tensor is None:
                self.zero_tensor = torch.zeros_like(tensor.mean(axis=0, keepdim=True)).repeat(
                    len(self.mean) - len(tensor), 1, 1)
            tensor = torch.concat([tensor, self.zero_tensor])

        return tensor


class OneHotEncode:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, tensor):
        return torch.nn.functional.one_hot(tensor, self.num_classes)
