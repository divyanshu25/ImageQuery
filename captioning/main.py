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
from collections import Counter

from architecture.encoder import EncoderCNN
from architecture.decoder import DecoderRNN
from captioning.captioning_config import CaptioningConfig
from data_handler.data_loader import get_data_loader
from data_handler.utils import parse_flickr
from train import train
from inference import get_predict
import torch.utils.data as data
import pprint

import torch.nn as nn
import torch

from utils import display_image


def execute():
    if CaptioningConfig.run_training:
        train_and_validate()
    if CaptioningConfig.run_prediction:
        predict()


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_stats(train_loader, val_loader):
    print("Total number of tokens in vocabulary:", len(train_loader.dataset.vocab))
    print("Total number of data points in train set:", len(train_loader.dataset))
    print("Total number of data points in val set:", len(val_loader.dataset))

    # print(dict(list(train_loader.dataset.vocab.word2idx.items())[:10]))
    # counter = Counter(train_loader.dataset.caption_lengths)
    # lengths = sorted(counter.items(), key=lambda pair: pair[1], reverse=True)
    # for value, count in lengths:
    #     print("value: %2d --- count: %5d" % (value, count))
    # indices = train_loader.dataset.get_train_indices()
    # # print("sampled indices:", indices)
    # new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
    # train_loader.batch_sampler.sampler = new_sampler
    #
    # # Obtain the batch.
    # images, captions = next(iter(train_loader))
    #
    # images, captions = images.to(device), captions.to(device)
    #
    # print("images.shape:", images.shape)
    # print("captions.shape:", captions.shape)
    # display_image(images, captions, train_loader)
    #
    return


def train_and_validate():
    # Step1: Load Data and Visulaize
    device = get_device()
    flickr_ann_dict = parse_flickr(CaptioningConfig.annotations_file)
    train_loader = get_data_loader(CaptioningConfig, flickr_ann_dict, mode="train")
    val_loader = get_data_loader(CaptioningConfig, flickr_ann_dict, mode="val")
    vocab_size = len(train_loader.dataset.vocab)
    # print_stats(train_loader, val_loader)

    # Step2: Define and Initialize Neural Net/ Model Class/ Hypothesis(H).
    encoder = EncoderCNN(CaptioningConfig.embed_size)
    encoder = encoder.to(device)
    decoder = DecoderRNN(
        CaptioningConfig.embed_size, CaptioningConfig.hidden_size, vocab_size
    )
    decoder = decoder.to(device)

    # Step3: Define Loss Function and optimizer
    params = list(decoder.parameters()) + list(encoder.resnet.fc.parameters())
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(params=params, lr=CaptioningConfig.learning_rate)
    # optimizer = torch.optim.SGD(
    #     params=params,
    #     lr=Config.learning_rate,
    #     momentum=Config.momentum,
    #     weight_decay=Config.weight_decay,
    # )

    # Step4: Train the network.
    if CaptioningConfig.load_from_file:
        encoder.load_state_dict(torch.load(CaptioningConfig.encoder_file))
        decoder.load_state_dict(torch.load(CaptioningConfig.decoder_file))

    train(encoder, decoder, optimizer, criterion, train_loader, val_loader, device)


def predict():
    flickr_ann_dict = parse_flickr(CaptioningConfig.annotations_file)
    test_loader = get_data_loader(CaptioningConfig, flickr_ann_dict, mode="test")
    vocab_size = len(test_loader.dataset.vocab)

    device = get_device()

    encoder = EncoderCNN(CaptioningConfig.embed_size)
    encoder = encoder.to(device)
    decoder = DecoderRNN(
        CaptioningConfig.embed_size, CaptioningConfig.hidden_size, vocab_size
    )
    decoder = decoder.to(device)

    encoder.eval()
    decoder.eval()
    if not torch.cuda.is_available():
        encoder.load_state_dict(
            torch.load(CaptioningConfig.encoder_file, map_location=torch.device("cpu"))
        )
        decoder.load_state_dict(
            torch.load(CaptioningConfig.decoder_file, map_location=torch.device("cpu"))
        )
    else:
        encoder.load_state_dict(torch.load(CaptioningConfig.encoder_file))
        decoder.load_state_dict(torch.load(CaptioningConfig.decoder_file))

    indices = test_loader.dataset.get_train_indices()
    new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
    test_loader.batch_sampler.sampler = new_sampler

    images, captions = next(iter(test_loader))
    images = images.to(device)
    # imshow(orig_image)
    get_predict(images, captions, encoder, decoder, test_loader)


if __name__ == "__main__":
    execute()
