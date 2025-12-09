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


import open_clip
import torch
import torch.nn as nn


def load_model(base_model: str, clone_weights: bool = True, channels: int = 12, ckpt_path: str = None, ):
    clip_model, _, _ = open_clip.create_model_and_transforms(base_model, ckpt_path)
    state_dict = clip_model.state_dict()
    orig_model = clip_model

        
    if clone_weights:
        clip_model.visual.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=orig_model.visual.conv1.out_channels,
            kernel_size=orig_model.visual.conv1.kernel_size,  # dynamically choose
            stride=orig_model.visual.conv1.stride,
            bias=orig_model.visual.conv1.bias
        )

        state_dict = extend_weights(state_dict=state_dict, channels=channels)

        clip_model.load_state_dict(state_dict)

    tokenizer = open_clip.get_tokenizer(base_model)

    return clip_model, tokenizer


#def extend_weights(state_dict: dict, channels: int):
    old_patch_weights = state_dict["visual.conv1.weight"]
    new_patch_weights = torch.zeros(
        (old_patch_weights.shape[0], channels, old_patch_weights.shape[2], old_patch_weights.shape[3]))
    new_patch_weights[:, 0:1, :, :] = old_patch_weights[:, 2:3, :, :]  # Keep original RGB weights but in BGR format
    new_patch_weights[:, 1:2, :, :] = old_patch_weights[:, 1:2, :, :]  # Keep original RGB weights
    new_patch_weights[:, 2:3, :, :] = old_patch_weights[:, 0:1, :, :]  # Keep original RGB weights
    state_dict["visual.conv1.weight"] = new_patch_weights

    return state_dict

# The initial function initialises randomly for rgb channels and 0 to other channels but we want everything to be random
def extend_weights(state_dict: dict, channels: int):
    old = state_dict["visual.conv1.weight"]          # (O, 3, kH, kW)
    O, _, kH, kW = old.shape
    new = old.new_empty(O, channels, kH, kW)         # allocate

    new[:, 0:1] = old[:, 2:3]
    new[:, 1:2] = old[:, 1:2]
    new[:, 2:3] = old[:, 0:1]

    torch.nn.init.kaiming_normal_(new[:, 3:], mode="fan_out", nonlinearity="relu")

    state_dict["visual.conv1.weight"] = new
    return state_dict
