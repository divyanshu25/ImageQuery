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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)

        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        captions = captions[:, :-1]
        embed = self.embedding_layer(captions)
        embed = torch.cat((features.unsqueeze(1), embed), dim=1)
        lstm_outputs, _ = self.lstm(embed)
        out = self.linear(lstm_outputs)

        return out


# class DecoderRNN(nn.Module):
#     def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
#         super(DecoderRNN, self).__init__()
#
#         # define the properties
#         self.embed_size = embed_size
#         self.hidden_size = hidden_size
#         self.vocab_size = vocab_size
#
#         # lstm cell
#         self.lstm_cell = nn.LSTMCell(input_size=embed_size, hidden_size=hidden_size)
#
#         # output fully connected layer
#         self.fc_out = nn.Linear(
#             in_features=self.hidden_size, out_features=self.vocab_size
#         )
#
#         # embedding layer
#         self.embed = nn.Embedding(
#             num_embeddings=self.vocab_size, embedding_dim=self.embed_size
#         )
#
#         # activations
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, features, captions):
#
#         # batch size
#         batch_size = features.size(0)
#
#         # init the hidden and cell states to zeros
#         hidden_state = torch.zeros((batch_size, self.hidden_size)).cuda()
#         cell_state = torch.zeros((batch_size, self.hidden_size)).cuda()
#
#         # define the output tensor placeholder
#         outputs = torch.empty((batch_size, captions.size(1), self.vocab_size)).cuda()
#
#         # embed the captions
#         captions_embed = self.embed(captions)
#
#         # pass the caption word by word
#         for t in range(captions.size(1)):
#
#             # for the first time step the input is the feature vector
#             if t == 0:
#                 hidden_state, cell_state = self.lstm_cell(
#                     features, (hidden_state, cell_state)
#                 )
#
#             # for the 2nd+ time step, using teacher forcer
#             else:
#                 hidden_state, cell_state = self.lstm_cell(
#                     captions_embed[:, t, :], (hidden_state, cell_state)
#                 )
#
#             # output of the attention mechanism
#             out = self.fc_out(hidden_state)
#
#             # build the output tensor
#             outputs[:, t, :] = out
#
#         return outputs
