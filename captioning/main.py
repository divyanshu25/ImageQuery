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
from captioning.captioning_config import Config as CaptioningConfig
from data_handler.data_loader import get_data_loader, get_vocabulary
from utils import convert_captions
from train import train
from inference import get_predict
import torch.nn as nn
import torch
from utils import display_image

config = CaptioningConfig()


def execute():
    if config.run_training:
        train_and_validate()
    if config.run_prediction:
        predict()


def get_device():
    return torch.cuda.is_available()


def print_stats(train_loader, val_loader):
    # print("Total number of tokens in vocabulary:", len(train_loader.dataset.vocab))
    # print("Total number of data points in train set:", len(train_loader.dataset))
    # print("Total number of data points in val set:", len(val_loader.dataset))

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
    train_loader = get_data_loader(config, mode="train", type=config.dataset_type)
    val_loader = get_data_loader(config, mode="val", type=config.dataset_type)
    vocab = get_vocabulary(config, type=config.dataset_type)
    vocab_size = len(vocab)
    # print_stats(train_loader, val_loader)
    # Step2: Define and Initialize Neural Net/ Model Class/ Hypothesis(H).
    encoder = EncoderCNN(config.embed_size)
    decoder = DecoderRNN(config.embed_size, config.hidden_size, vocab_size)
    criterion = nn.CrossEntropyLoss()

    if device:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        criterion = criterion.cuda()
    # Step3: Define Loss Function and optimizer

    params = None
    if config.train_encoder:
        params = list(decoder.parameters()) + list(encoder.parameters())
    else:
        params = list(decoder.parameters()) + list(encoder.resnet.fc.parameters())

    optimizer = torch.optim.Adam(params=params, lr=config.learning_rate)

    # Step4: Train the network.
    if config.load_from_file:
        print(
            "Loading encoder from {} and decoder from {} to resume training.".format(
                config.encoder_file, config.decoder_file
            )
        )
        if not torch.cuda.is_available():
            encoder.load_state_dict(
                torch.load(config.encoder_file, map_location=torch.device("cpu"))
            )
            decoder.load_state_dict(
                torch.load(config.decoder_file, map_location=torch.device("cpu"))
            )
        else:
            encoder.load_state_dict(torch.load(config.encoder_file))
            decoder.load_state_dict(torch.load(config.decoder_file))

    train(
        encoder, decoder, optimizer, criterion, train_loader, val_loader, device, vocab
    )


def predict():
    test_loader = get_data_loader(config, mode="test", type=config.dataset_type)
    vocab = get_vocabulary(config, type=config.dataset_type)
    vocab_size = len(vocab)

    device = get_device()

    encoder = EncoderCNN(config.embed_size)
    decoder = DecoderRNN(config.embed_size, config.hidden_size, vocab_size)
    if device:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    encoder.eval()
    decoder.eval()
    if not torch.cuda.is_available():
        encoder.load_state_dict(
            torch.load(config.encoder_file, map_location=torch.device("cpu"))
        )
        decoder.load_state_dict(
            torch.load(config.decoder_file, map_location=torch.device("cpu"))
        )
    else:
        encoder.load_state_dict(torch.load(config.encoder_file))
        decoder.load_state_dict(torch.load(config.decoder_file))

    images, captions = convert_captions(next(iter(test_loader)), vocab, config)
    if device:
        images = images.cuda()
    get_predict(images, encoder, decoder, vocab, captions)


if __name__ == "__main__":
    execute()
