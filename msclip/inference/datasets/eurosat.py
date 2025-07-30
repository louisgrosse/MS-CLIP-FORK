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
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

from msclip.inference.datasets.transforms import ToTensor


class EuroSATRGB(ImageFolder):
    """ Sentinel-2 RGB Land Cover Classification dataset from 'EuroSAT: A Novel Dataset
    and Deep Learning Benchmark for Land Use and Land Cover Classification', Helber at al. (2017)
    https://arxiv.org/abs/1709.00029

    'We present a novel dataset based on Sentinel-2 satellite images covering 13 spectral
    bands and consisting out of 10 classes with in total 27,000 labeled and geo-referenced images.'

    Note: RGB bands only
    """

    def __init__(
            self,
            root: str = "/",
            transform: T.Compose = T.Compose([T.ToTensor()])
    ):
        super().__init__(
            root=os.path.join(root),
            transform=transform
        )


class EuroSATMS(ImageFolder):
    """ Sentinel-2 RGB Land Cover Classification dataset from 'EuroSAT: A Novel Dataset
    and Deep Learning Benchmark for Land Use and Land Cover Classification', Helber at al. (2017)
    https://arxiv.org/abs/1709.00029

    'We present a novel dataset based on Sentinel-2 satellite images covering 13 spectral
    bands and consisting out of 10 classes with in total 27,000 labeled and geo-referenced images.'

    Note: all 13 multispectral (MS) bands
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
    # root=os.path.join(root, "EuroSAT/S2L2A")
