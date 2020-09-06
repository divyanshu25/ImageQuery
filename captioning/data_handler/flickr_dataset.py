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

    def getIds(self, id_file):
        ids = []
        with open(id_file, "r") as f:
            ids = f.read().splitlines()
        return ids

    def __init__(
        self,
        img_dir,
        id_file,
        vocab,
        mode="train",
        ann_dict=None,
        transform=None,
        target_transform=None,
    ):
        super().__init__()
        self.id_file = id_file
        self.img_dir = img_dir
        self.ids = self.getIds(id_file)
        self.vocab = vocab
        self.target_transform = target_transform
        self.transform = transform
        self.mode = mode
        self.ann_dict = ann_dict
        print("Getting tokens from all captions to generate length...")
        all_tokens = [
            nltk.tokenize.word_tokenize(
                str(self.ann_dict[self.ids[index]]).lower()
            )
            for index in tqdm(np.arange(len(self.ids)))
        ]
        self.caption_lengths = [len(token) for token in all_tokens]

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

        # Captions
        if self.mode == "train" or self.mode == "val":
            caption = []
            target = self.ann_dict[img_id]
            target = str(" ".join(target))
            tokens = nltk.tokenize.word_tokenize(target.lower())
            caption.append(self.vocab(self.vocab.start_word))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.vocab.end_word))
            # caption = caption[:1]
            caption = torch.Tensor(caption).long()

            return img, caption
        else:
            orig_image = np.array(img)
            image = orig_image.copy()
            if self.transform is not None:
                image = self.transform(img)
            # return original image and pre-processed image tensor
            return orig_image, image

    def __len__(self):
        return len(self.ids)
