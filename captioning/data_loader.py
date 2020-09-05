#  ================================================================
#  Copyright [2020] [Divyanshu Goyal]
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==================================================================

from torch.utils.data import Dataset, DataLoader, sampler
from pathlib import Path
from PIL import Image
import os
import matplotlib.pyplot as plt


class Flickr8kCustom(Dataset):
    """
    Args:
        root (string): Root directory where images are downloaded to.
        ann_file (string): Path to annotation file.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def getIds(self, id_file):
        ids = []
        with open(id_file, "r") as f:
            ids = f.read().splitlines()
        return ids

    def __init__(
        self, img_dir, id_file, annotations, transform=None, target_transform=None
    ):
        super().__init__()
        self.id_file = id_file
        self.img_dir = img_dir
        self.ids = self.getIds(id_file)
        self.annotations = annotations
        self.target_transform = target_transform
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        img_id = self.ids[index]

        # Image
        img_path = os.path.join(self.img_dir, img_id)
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # Captions
        target = self.annotations[img_id]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, ';'.join(target)

    def __len__(self):
        return len(self.ids)
