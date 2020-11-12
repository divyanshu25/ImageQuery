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
from .modules import *

config = Config()

class DecoderKWTL(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(DecoderKWTL,self).__init__()
        att_dim = config.decoderkwtl_attdim
        encoded_dim = config.decoderkwtl_encodeddim
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.encoded_to_hidden = nn.Linear(encoded_dim, hidden_size)
        self.global_features = nn.Linear(encoded_dim, embed_size)
        self.LSTM = AdaptiveLSTMCell(embed_size * 2, hidden_size)
        self.adaptive_attention = AdaptiveAttention(hidden_size, att_dim)
        # input to the LSTMCell should be of shape (batch, input_size). Remember we are concatenating the word with
        # the global image features, therefore out input features should be embed_size * 2
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(p=0.5)
        self.init_weights()
        self.cuda_available = False
        if torch.cuda.is_available():
            self.cuda_available = True;

    def init_weights(self):
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.embedding.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, enc_image):
        h = torch.zeros(enc_image.shape[0], 512)
        c = torch.zeros(enc_image.shape[0], 512)
        if self.cuda_available:
            h = h.cuda()
            c = c.cuda()
        return h, c

    def forward(self, image_features, encoded_captions, caption_lengths):

        """
        enc_image: the encoded images from the encoder, of shape (batch_size, num_pixels, 2048)
        global_features: the global image features returned by the Encoder, of shape: (batch_size, 2048)
        encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        """
        enc_image, global_features = image_features
        spatial_image = F.relu(self.encoded_to_hidden(enc_image))  # (batch_size,num_pixels,hidden_size)
        global_image = F.relu(self.global_features(global_features))      # (batch_size,embed_size)
        batch_size = spatial_image.shape[0]
        num_pixels = spatial_image.shape[1]
        # Sort input data by decreasing lengths
        # caption_lenghts will contain the sorted lengths, and sort_ind contains the sorted elements indices
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        #The sort_ind contains elements of the batch index of the tensor encoder_out. For example, if sort_ind is [3,2,0],
        #then that means the descending order starts with batch number 3,then batch number 2, and finally batch number 0.
        spatial_image = spatial_image[sort_ind]           # (batch_size,num_pixels,hidden_size) with sorted batches
        global_image = global_image[sort_ind]             # (batch_size, embed_size) with sorted batches
        encoded_captions = encoded_captions[sort_ind]     # (batch_size, max_caption_length) with sorted batches
        enc_image = enc_image[sort_ind]                   # (batch_size, num_pixels, 2048)

        # Embedding. Each batch contains a caption. All batches have the same number of rows (words), since we previously
        # padded the ones shorter than max_caption_length, as well as the same number of columns (embed_dim)
        embeddings = self.embedding(encoded_captions)     # (batch_size, max_caption_length, embed_dim)

        # Initialize the LSTM state
        h, c = self.init_hidden_state(enc_image)          # (batch_size, hidden_size)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # decode_lengths = (caption_lengths - 1).tolist()
        decode_lengths = (caption_lengths).tolist()

        # Create tensors to hold word predicion scores,alphas and betas
        predictions = torch.zeros(batch_size, max(decode_lengths), self.vocab_size)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels+1)
        betas = torch.zeros(batch_size, max(decode_lengths),1)

        if self.cuda_available:
            predictions = predictions.cuda()
            alphas = alphas.cuda()
            betas = betas.cuda()

        # Concatenate the embeddings and global image features for input to LSTM
        global_image = global_image.unsqueeze(1).expand_as(embeddings)
        inputs = torch.cat((embeddings,global_image), dim = 2)    # (batch_size, max_caption_length, embed_dim * 2)

        #Start decoding
        for timestep in range(max(decode_lengths)):
            # Create a Packed Padded Sequence manually, to process only the effective batch size N_t at that timestep. Note
            # that we cannot use the pack_padded_seq provided by torch.util because we are using an LSTMCell, and not an LSTM
            batch_size_t = sum([l > timestep for l in decode_lengths])
            current_input = inputs[:batch_size_t, timestep, :]             # (batch_size_t, embed_dim * 2)
            h, c, st = self.LSTM(current_input, (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, hidden_size)
            # Run the adaptive attention model
            out_l, alpha_t, beta_t = self.adaptive_attention(spatial_image[:batch_size_t],h,st)
            # Compute the probability over the vocabulary
            pt = self.fc(self.dropout(out_l))                  # (batch_size, vocab_size)
            predictions[:batch_size_t, timestep, :] = pt
            alphas[:batch_size_t, timestep, :] = alpha_t
            betas[:batch_size_t, timestep, :] = beta_t
        return predictions#, alphas, betas, encoded_captions, decode_lengths, sort_ind
