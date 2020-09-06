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
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, sampler
from pathlib import Path
from PIL import Image
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from captioning.data_handler.flickr_dataset import Flickr8kCustom
from captioning.data_handler.vocabulary import Vocabulary
from captioning.data_handler.utils import parse_flickr


def get_data_loader(config, flickr_ann_dict, mode="train"):

    """Load and normalizing the CIFAR10 training and test datasets using torchvision"""

    transform = transforms.Compose(
        [
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    vocab = Vocabulary(
        config.vocab_threshold,
        config.vocab_file,
        config.annotations_file,
        vocab_from_file=True,
    )

    dataset = None
    data_loader = None
    if mode == "train":
        dataset = Flickr8kCustom(
            img_dir=config.images_dir,
            id_file=config.train_id_file,
            mode="train",
            batch_size=config.batch_size,
            vocab=vocab,
            ann_dict=flickr_ann_dict,
            transform=transform,
        )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=True,
        )

    elif mode == "test":
        dataset = Flickr8kCustom(
            img_dir=config.images_dir,
            id_file=config.test_id_file,
            mode="test",
            batch_size=config.batch_size,
            vocab=vocab,
            transform=transform,
        )
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=True,
        )
    elif mode == "val":
        dataset = Flickr8kCustom(
            img_dir=config.images_dir,
            id_file=config.val_id_file,
            mode="val",
            batch_size=config.batch_size,
            vocab=vocab,
            ann_dict=flickr_ann_dict,
            transform=transform,
        )
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=True,
        )

    return data_loader
