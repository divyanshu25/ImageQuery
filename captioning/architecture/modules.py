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
from torch.nn import init
import torch.nn.functional as F


class AdaptiveLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AdaptiveLSTMCell, self).__init__()
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.x_gate = nn.Linear(input_size, hidden_size)
        self.h_gate = nn.Linear(hidden_size, hidden_size)

    def forward(self, inp, states):
        h_old, c_old = states
        ht, ct = self.lstm_cell(inp, (h_old, c_old))
        sen_gate = F.sigmoid(self.x_gate(inp) + self.h_gate(h_old))
        st = sen_gate * F.tanh(ct)
        return ht, ct, st


class AdaptiveAttention(nn.Module):
    def __init__(self, hidden_size, att_dim):
        super(AdaptiveAttention, self).__init__()
        self.sen_affine = nn.Linear(hidden_size, hidden_size)
        self.sen_att = nn.Linear(hidden_size, att_dim)
        self.h_affine = nn.Linear(hidden_size, hidden_size)
        self.h_att = nn.Linear(hidden_size, att_dim)
        self.v_att = nn.Linear(hidden_size, att_dim)
        self.alphas = nn.Linear(att_dim, 1)
        self.context_hidden = nn.Linear(hidden_size, hidden_size)

    def forward(self, spatial_image, decoder_out, st):
        """
        spatial_image: the spatial image of size (batch_size,num_pixels,hidden_size)
        decoder_out: the decoder hidden state of shape (batch_size, hidden_size)
        st: visual sentinal returned by the Sentinal class, of shape: (batch_size, hidden_size)
        """
        num_pixels = spatial_image.shape[1]
        visual_attn = self.v_att(spatial_image)  # (batch_size,num_pixels,att_dim)
        sentinel_affine = F.relu(self.sen_affine(st))  # (batch_size,hidden_size)
        sentinel_attn = self.sen_att(sentinel_affine)  # (batch_size,att_dim)

        hidden_affine = F.tanh(self.h_affine(decoder_out))  # (batch_size,hidden_size)
        hidden_attn = self.h_att(hidden_affine)  # (batch_size,att_dim)

        hidden_resized = hidden_attn.unsqueeze(1).expand(
            hidden_attn.size(0), num_pixels + 1, hidden_attn.size(1)
        )

        concat_features = torch.cat(
            [spatial_image, sentinel_affine.unsqueeze(1)], dim=1
        )  # (batch_size, num_pixels+1, hidden_size)
        attended_features = torch.cat(
            [visual_attn, sentinel_attn.unsqueeze(1)], dim=1
        )  # (batch_size, num_pixels+1, att_dim)

        attention = F.tanh(
            attended_features + hidden_resized
        )  # (batch_size, num_pixels+1, att_dim)

        alpha = self.alphas(attention).squeeze(2)  # (batch_size, num_pixels+1)
        att_weights = F.softmax(alpha, dim=1)  # (batch_size, num_pixels+1)

        context = (concat_features * att_weights.unsqueeze(2)).sum(
            dim=1
        )  # (batch_size, hidden_size)
        beta_value = att_weights[:, -1].unsqueeze(1)  # (batch_size, 1)

        out_l = F.tanh(self.context_hidden(context + hidden_affine))

        return out_l, att_weights, beta_value


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(
            encoder_dim, attention_dim
        )  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(
            decoder_dim, attention_dim
        )  # linear layer to transform decoder's output
        self.full_att = nn.Linear(
            attention_dim, 1
        )  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(
            2
        )  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(
            dim=1
        )  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha
