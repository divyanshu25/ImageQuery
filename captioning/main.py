#  ================================================================
#  Copyright 2020 Image Query Team
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==================================================================
from collections import Counter
from captioning.architecture.archpicker import get_encoder_decoder
from captioning.captioning_config import Config as CaptioningConfig
from captioning.data_handler.data_loader import get_data_loader, get_vocabulary
from captioning.utils import convert_captions
from captioning.train import train
from captioning.inference import get_predict
import torch.nn as nn
import torch
from bert.bert_encoder import BERT

# from utils import display_image

config = CaptioningConfig()
bert = BERT()


def execute():
    if config.run_training:
        train_and_validate()
    if config.run_prediction:
        predict()


def get_device():
    return torch.cuda.is_available()


def train_and_validate():
    # Step1: Load Data and Visulaize
    device = get_device()
    train_loader = get_data_loader(config, mode="train", type=config.dataset_type)
    val_loader = get_data_loader(config, mode="val", type=config.dataset_type)
    vocab = get_vocabulary(config, type=config.dataset_type, bert=bert)
    vocab_size = len(vocab)
    print("Vocab Size: ", vocab_size)
    # print_stats(train_loader, val_loader)
    # Step2: Define and Initialize Neural Net/ Model Class/ Hypothesis(H).
    encoder, decoder = get_encoder_decoder(
        config.embed_size, config.hidden_size, vocab_size, bert
    )
    criterion = nn.CrossEntropyLoss()
    if device:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        criterion = criterion.cuda()
    # Step3: Define Loss Function and optimizer

    params = list(decoder.parameters()) + encoder.get_learning_parameters()

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
        encoder,
        decoder,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        device,
        vocab,
        bert,
    )


def predict():
    test_loader = get_data_loader(config, mode="test", type=config.dataset_type)
    vocab = get_vocabulary(config, type=config.dataset_type, bert=bert)
    vocab_size = len(vocab)

    device = get_device()

    encoder, decoder = get_encoder_decoder(
        config.embed_size, config.hidden_size, vocab_size, bert=bert
    )
    if device:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    encoder.eval()
    decoder.eval()
    if not torch.cuda.is_available():
        print(config.encoder_file, config.decoder_file)
        encoder.load_state_dict(
            torch.load(config.encoder_file, map_location=torch.device("cpu"))
        )
        decoder.load_state_dict(
            torch.load(config.decoder_file, map_location=torch.device("cpu"))
        )
    else:
        encoder.load_state_dict(torch.load(config.encoder_file))
        decoder.load_state_dict(torch.load(config.decoder_file))

    images, captions, _ = next(iter(test_loader))
    images, captions, captions_length = convert_captions(
        images, captions, vocab, config, bert=bert
    )
    if device:
        images = images.cuda()
    get_predict(images, encoder, decoder, vocab, captions=captions, bert=bert)


if __name__ == "__main__":
    execute()
