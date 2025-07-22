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
import tifffile
from PIL import Image
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from src.inference.datasets.transforms import ToTensor


class METERML(ImageFolder):
    """ METERML
    """

    def __init__(
            self,
            root: str = "./",
            transform: T.Compose = T.Compose([ToTensor()])
    ):
        super().__init__(
            root=os.path.join(root),
            transform=transform,
            loader=tifffile.imread
        )


class METERML_NAIP(ImageFolder):
    """ METERML
    """

    def __init__(
            self,
            root: str = "./",
            transform: T.Compose = T.Compose([ToTensor()])
    ):
        super().__init__(
            root=root,
            transform=transform,
            loader=Image.open
        )
