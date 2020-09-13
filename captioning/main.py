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


from captioning.architecture.encoder import EncoderCNN
from captioning.architecture.decoder import DecoderRNN
from captioning.config import Config
from captioning.data_handler.data_loader import get_data_loader
from captioning.utils import display_image
from captioning.data_handler.utils import parse_flickr
from captioning.train import train
from captioning.inference import get_predict
import numpy as np
import torch.utils.data as data

import torch.nn as nn
import torch
import math
import sys
import os


def execute():
    if Config.run_training:
        train_and_validate()
    if Config.run_prediction:
        predict()


def train_and_validate():
    # Step1: Load Data
    flickr_ann_dict = parse_flickr(Config.annotations_file)
    train_loader = get_data_loader(Config, flickr_ann_dict, mode="train")
    val_loader = get_data_loader(Config, flickr_ann_dict, mode="val")
    vocab_size = len(train_loader.dataset.vocab)
    # print("Total number of tokens in vocabulary:", len(train_loader.dataset.vocab))
    # print("Total number of tokens in vocabulary:", len(train_loader.dataset.vocab))
    # print(train_loader.dataset.vocab("ieowoqjf"))
    # counter = Counter(train_loader.dataset.caption_lengths)
    # lengths = sorted(counter.items(), key=lambda pair: pair[1], reverse=True)
    # for value, count in lengths:
    #     print("value: %2d --- count: %5d" % (value, count))

    indices = train_loader.dataset.get_train_indices()
    print("sampled indices:", indices)
    new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
    train_loader.batch_sampler.sampler = new_sampler

    # Obtain the batch.
    images, captions = next(iter(train_loader))

    print("images.shape:", images.shape)
    print("captions.shape:", captions.shape)
    # display_image(train_loader)

    # Step2: Define and Initialize Neural Net/ Model Class/ Hypothesis(H).
    encoder = EncoderCNN(Config.embed_size)
    decoder = DecoderRNN(Config.embed_size, Config.hidden_size, vocab_size)

    # features = encoder(images)
    #
    # print('type(features):', type(features))
    # print('features.shape:', features.shape)
    #
    # outputs = decoder(features, captions)
    #
    # print('type(outputs):', type(outputs))
    # print('outputs.shape:', outputs.shape)
    # Step3: Define Loss Function and optimizer
    params = list(decoder.parameters()) + list(encoder.embed.parameters())
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        params=params, lr=Config.learning_rate, momentum=Config.momentum
    )

    # Step4: Train the network.
    if Config.load_from_file:
        encoder.load_state_dict(torch.load(Config.encoder_file))
        decoder.load_state_dict(torch.load(Config.decoder_file))

    train(encoder, decoder, optimizer, criterion, train_loader, val_loader)

    #
    # # Step5: Save Model
    # PATH = "./model/cifar_net.pth"
    # # torch.save(net.state_dict(), PATH)
    #
    # # Step6: Test the net on Test Data
    # net = Net()
    # net.load_state_dict(torch.load(PATH))
    # run_test(test_loader, net, classes


def predict():
    flickr_ann_dict = parse_flickr(Config.annotations_file)
    test_loader = get_data_loader(Config, flickr_ann_dict, mode="test")
    vocab_size = len(test_loader.dataset.vocab)

    encoder = EncoderCNN(Config.embed_size)
    encoder.eval()
    decoder = DecoderRNN(Config.embed_size, Config.hidden_size, vocab_size)
    decoder.eval()

    encoder.load_state_dict(torch.load(Config.encoder_file))
    decoder.load_state_dict(torch.load(Config.decoder_file))

    indices = test_loader.dataset.get_train_indices()
    # print("sampled indices:", indices)
    new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
    test_loader.batch_sampler.sampler = new_sampler

    orig_image, image = next(iter(test_loader))
    get_predict(image, encoder, decoder, test_loader)


if __name__ == "__main__":
    execute()
