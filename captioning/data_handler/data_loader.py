#   ================================================================
#   Copyright [2020] [Image Query Team]
#  #
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#  #
#       http://www.apache.org/licenses/LICENSE-2.0
#  #
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#   ==================================================================

import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, sampler, Subset
from torchvision import transforms
import torchvision.datasets as dset
from torch.utils.data._utils.collate import default_collate

from captioning.data_handler.flickr_dataset import Flickr8kCustom
from captioning.data_handler.vocabulary import Vocabulary
from captioning.data_handler.utils import parse_flickr, parse_coco


def get_data_loader(config, mode="train", type="flickr8k"):
    data_loader = None

    if type == "flickr8k":
        flickr_ann_dict = parse_flickr(config.annotations_file)
        data_loader = get_flickr_data_loader(config, flickr_ann_dict, mode)
    elif type == "coco":
        data_loader = get_coco_data_loader(config, mode)
    else:
        print("Wrong dataset type received : " + type)

    return data_loader


def get_vocabulary(config, type="flickr8k"):
    vocab = None

    if type == "flickr8k":
        vocab = Vocabulary(
            config.vocab_threshold,
            config.vocab_file,
            vocab_from_file=config.vocab_from_file,
        )
        if not (os.path.exists(config.vocab_file) and config.vocab_from_file):
            ann_dict = parse_flickr(config.annotations_file)
            vocab.add_captions(ann_dict)
            vocab.dump_vocab_in_file()

    elif type == "coco":
        vocab = Vocabulary(
            config.vocab_threshold,
            config.vocab_file,
            vocab_from_file=config.vocab_from_file,
        )
        if not (os.path.exists(config.vocab_file) and config.vocab_from_file):
            vocab.add_captions(parse_coco(config.train_ann_file_coco))
            vocab.add_captions(parse_coco(config.val_ann_file_coco))
            vocab.dump_vocab_in_file()
    else:
        print("Wrong dataset type received : " + type)

    return vocab


def get_flickr_data_loader(config, flickr_ann_dict, mode="train"):

    """Load and normalizing the CIFAR10 training and test datasets using torchvision"""

    transform = transforms.Compose(
        [
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    dataset = None
    data_loader = None
    if mode == "train":
        dataset = Flickr8kCustom(
            img_dir=config.images_dir,
            id_file=config.train_id_file,
            mode="train",
            batch_size=config.batch_size,
            ann_dict=flickr_ann_dict,
            transform=transform,
        )
        if config.flickr_subsample:
            indices = range(0, config.flickr_subset_size_train)
            dataset = Subset(dataset, indices)

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
            ann_dict=flickr_ann_dict,
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
            ann_dict=flickr_ann_dict,
            transform=transform,
        )
        if config.flickr_subsample:
            indices = range(0, config.flickr_subset_size_val)
            dataset = Subset(dataset, indices)

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=True,
        )

    return data_loader


def get_coco_data_loader(config, mode="train"):
    def collate_pad(data):
        new_data = []
        for d in data:
            captions = d[1]
            new_captions = []
            for c in captions:
                for i in range(config.max_char_length - len(c)):
                    c += "."
                c = c[0 : config.max_char_length - 1]
                c += "."
                new_captions.append(c)
            new_d = (d[0], new_captions, d[2])
            new_data.append(new_d)
        return default_collate(new_data)

    transform = transforms.Compose(
        [
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    dataset = None

    if mode == "train":
        trainset = dset.CocoCaptions(
            root=config.train_root_dir_coco,
            annFile=config.train_ann_file_coco,
            transform=transform,
        )
        indices = range(0, config.coco_subset_size)
        dataset = Subset(trainset, indices)

    elif mode == "val":
        dataset = dset.CocoCaptions(
            root=config.val_root_dir_coco,
            annFile=config.val_ann_file_coco,
            transform=transform,
        )

    elif mode == "test":
        dataset = dset.CocoCaptions(
            root=config.test_root_dir_coco,
            annFile=config.test_ann_file_coco,
            transform=transform,
        )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        collate_fn=collate_pad,
    )

    return data_loader
