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

import torchvision.transforms as T
from torchvision.datasets import ImageFolder


class AID(ImageFolder):
    """ Aerial Image Dataset (AID) from 'AID: A Benchmark Dataset for Performance
    Evaluation of Aerial Scene Classification', Xia et al. (2017)
    https://arxiv.org/abs/1608.05167

    'The AID dataset has a number of 10000 images within 30 classes.'
    """

    def __init__(
            self,
            root: str = "/",
            transform: T.Compose = T.Compose([T.ToTensor()])
    ):
        super().__init__(
            root=root,
            transform=transform
        )
