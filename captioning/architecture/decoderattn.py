# ================================================================
# Copyright 2020 Image Query Team
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
# ==================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from captioning_config import Config as Config
from .attention import *

config = Config()


class DecoderAttn(nn.Module):
    """
    Decoder with Attention.
    Reference Paper: Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
    Link: https://arxiv.org/abs/1502.03044
    """

    def __init__(self, embed_size, decoder_size, vocab_size):
        super(DecoderAttn, self).__init__()

        self.encoder_size, self.attn_size, self.embed_size, self.decoder_size, self.vocab_size, self.dropout = (
            config.features_img_size,
            config.attn_size,
            embed_size,
            decoder_size,
            vocab_size,
            config.dropout,
        )

        self.attn = Attention(self.encoder_size, decoder_size, self.attn_size)

        self.lstm_cell = nn.LSTMCell(
            embed_size + self.encoder_size, decoder_size, bias=True
        )

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embedding.weight.data.uniform_(-0.1, 0.1)

        self.dropout = nn.Dropout(p=self.dropout)

        self.h0 = nn.Linear(self.encoder_size, decoder_size)
        self.c0 = nn.Linear(self.encoder_size, decoder_size)
        self.f_beta = nn.Linear(decoder_size, self.encoder_size)
        self.sigmoid = nn.Sigmoid()

        self.fc = nn.Linear(decoder_size, vocab_size)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def initialize_prediction(self, shape, mcl, vs):
        p = torch.zeros(shape[0], mcl, vs)
        a = torch.zeros(shape[0], mcl, shape[1])
        return p, a

    def forward(self, features, captions, caption_lengths):
        features = features.view(features.shape[0], -1, self.encoder_size)

        caption_lengths, argsort_ind = torch.sort(caption_lengths.squeeze(1), dim = 0, descending=True)

        features = features[argsort_ind]
        captions = captions[argsort_ind]

        embeddings = self.embedding(captions)

        eo = features.mean(dim=1)
        h = self.h0(eo);
        c = self.c0(eo);

        decode_lengths = (caption_lengths - 1).tolist()
        mcl = max(decode_lengths)
        predictions, alphas = self.initialize_prediction(features.shape, mcl, self.vocab_size)

        for t in range(mcl):
            mask = sum([l > t for l in decode_lengths])
            attn_weighted, alpha = self.attn(
                features[:mask], h[:mask]
            )
            attn_weighted = self.sigmoid(self.f_beta(h[:mask])) * attn_weighted
            h, c = self.lstm_cell(
                torch.cat(
                    [embeddings[:mask, t, :], attn_weighted],
                    dim=1,
                ),
                (h[:mask], c[:mask]),
            )
            preds = self.fc(self.dropout(h))
            predictions[:mask, t, :] = preds
            alphas[:mask, t, :] = alpha

        return predictions, captions, decode_lengths, alphas

    def init_search(self, inputs, states=None, max_len=20):
        batch_size = inputs.size(0)
        encoder_size = inputs.size(-1)

        encoder_out = inputs.view(batch_size, -1, encoder_size)

        eo = encoder_out.mean(dim=1)
        h = self.h0(eo);
        c = self.c0(eo);

        return (encoder_out, (h, c))

    def predict_next(self, encoder_out, current_word, state):
        current_h, current_s = state
        embeddings = self.embedding(current_word).squeeze(1)
        att, _ = self.attn(encoder_out, current_h)
        gate = self.sigmoid(self.f_beta(current_h))
        att = gate * att
        current_h, current_s = self.lstm_cell(
            torch.cat([embeddings, att], dim=1), (current_h, current_s)
        )
        scores = self.fc(current_h)
        return scores, (current_h, current_s)
