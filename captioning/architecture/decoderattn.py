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
    Decoder.
    """

    def __init__(
        self, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5
    ):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderAttn, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = config.decoderattn_attdim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(
            encoder_dim, decoder_dim, self.attention_dim
        )  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(
            embed_dim + encoder_dim, decoder_dim, bias=True
        )  # decoding LSTMCell
        self.init_h = nn.Linear(
            encoder_dim, decoder_dim
        )  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(
            encoder_dim, decoder_dim
        )  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(
            decoder_dim, encoder_dim
        )  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(
            decoder_dim, vocab_size
        )  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(
            batch_size, -1, encoder_dim
        )  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(
            dim=0, descending=True
        )
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(
            encoded_captions
        )  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()
        # decode_lengths = (caption_lengths).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(
                encoder_out[:batch_size_t], h[:batch_size_t]
            )
            gate = self.sigmoid(
                self.f_beta(h[:batch_size_t])
            )  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat(
                    [embeddings[:batch_size_t, t, :], attention_weighted_encoding],
                    dim=1,
                ),
                (h[:batch_size_t], c[:batch_size_t]),
            )  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

    def sample(self, inputs, states=None, max_len=20):
        """ accepts pre-processed image tensor (inputs) and
        returns predicted sentence (list of tensor ids of length max_len) """
        # input 1,300
        batch_size = inputs.size(0)  # 1
        encoder_dim = inputs.size(-1)  # 2048

        # Flatten image
        encoder_out = inputs.view(
            batch_size, -1, encoder_dim
        )  # (batch_size, num_pixels, encoder_dim)

        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        beam_size = config.beam_size
        sequences = [
            [1.0, torch.LongTensor([[0]]), [0], (h, c)]
        ]  # [Value, curr_word, output_sentence, states]
        finished_beams = []
        best_so_far = 0.0

        for i in range(max_len):
            expanded_beams = []
            for index, s in enumerate(sequences):
                current_h, current_s = s[3]
                embeddings = self.embedding(s[1]).squeeze(1)
                att, _ = self.attention(
                    encoder_out, current_h
                )  # (s, encoder_dim), (s, num_pixels)
                gate = self.sigmoid(
                    self.f_beta(current_h)
                )  # gating scalar, (s, encoder_dim)
                att = gate * att
                current_h, current_s = self.decode_step(
                    torch.cat([embeddings, att], dim=1), (current_h, current_s)
                )  # (s, decoder_dim)
                scores = self.fc(current_h)
                out = F.softmax(scores, dim=1)
                topk_picks = torch.topk(out, beam_size, dim=1)  #
                topk_picks_values = topk_picks[0].squeeze()
                topk_picks_indices = topk_picks[1].squeeze()
                for ix, val in zip(topk_picks_indices, topk_picks_values):
                    current_beam = []
                    current_beam.extend(
                        [
                            s[0] * val.item(),
                            torch.LongTensor([ix]),
                            s[2] + [ix.item()],
                            (current_h, current_s),
                        ]
                    )
                    if ix.item() == 1:
                        finished_beams.append(current_beam)
                        if best_so_far < current_beam[0]:
                            best_so_far = current_beam[0]
                    else:
                        expanded_beams.append(current_beam)

            ordered = sorted(expanded_beams, key=lambda tup: tup[0])[::-1]
            sequences = ordered[:beam_size]

        sequences.extend(finished_beams)
        ordered = sorted(sequences, key=lambda tup: tup[0])[::-1]
        output_sentences = []
        for beam in ordered[:beam_size]:
            output_sentences.append(beam[2])
        return output_sentences
