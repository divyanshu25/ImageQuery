import torch
from torch.utils.data import Dataset, DataLoader, sampler
from pathlib import Path
from PIL import Image
import os
import nltk
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


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

    def __init__(
        self,
        img_dir,
        id_file,
        vocab,
        mode="train",
        batch_size=1,
        ann_dict=None,
        transform=None,
        target_transform=None,
    ):
        super().__init__()
        self.id_file = id_file
        self.img_dir = img_dir
        self.ids = self.get_ids(id_file)
        self.vocab = vocab
        self.target_transform = target_transform
        self.transform = transform
        self.mode = mode
        self.ann_dict = ann_dict
        self.batch_size = batch_size
        print("Getting tokens from all captions to generate length...")
        all_tokens = [
            nltk.tokenize.word_tokenize(str(self.ann_dict[self.ids[index]]).lower())
            for index in tqdm(np.arange(len(self.ids)))
        ]
        self.caption_lengths = [len(token) for token in all_tokens]

    def get_ids(self, id_file):
        ids = []
        new_ids = []
        with open(id_file, "r") as f:
            ids = f.read().splitlines()
        for id in ids:
            new_ids.extend([id+"#0", id+"#1", id+"#2", id+"#3", id+"#4"])
        return new_ids

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        img_id = self.ids[index]

        # Image
        img_path = os.path.join(self.img_dir, img_id[:-2])
        img = Image.open(img_path).convert("RGB")

        # Captions
        if self.mode == "train" or self.mode == "val":
            caption = []
            target = self.ann_dict[img_id]
            tokens = nltk.tokenize.word_tokenize(str(target).lower())
            caption.append(self.vocab(self.vocab.start_word))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.vocab.end_word))
            # caption = caption[:1]
            caption = torch.Tensor(caption).long()
            if self.transform is not None:
                img = self.transform(img)

            return img, caption
        else:
            orig_image = np.array(img)
            image = orig_image.copy()
            if self.transform is not None:
                image = self.transform(img)
            # return original image and pre-processed image tensor
            return orig_image, image

    def get_train_indices(self):
        sel_length = np.random.choice(self.caption_lengths)
        all_indices = np.where(
            [
                self.caption_lengths[i] == sel_length
                for i in np.arange(len(self.caption_lengths))
            ]
        )[0]
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

    def __len__(self):
        return len(self.ids)
