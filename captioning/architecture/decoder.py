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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)

        self.project_feature_layer = nn.Linear(embed_size, hidden_size)

        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        captions = captions[:, :-1]
        # print(captions.shape)
        embed = self.embedding_layer(captions)
        # print(embed.shape)
        embed = torch.cat((features.unsqueeze(1), embed), dim=1)
        # print(embed.shape)
        lstm_outputs, _ = self.lstm(embed)
        # print(lstm_outputs.shape)
        out = self.linear(lstm_outputs)
        # print(out.shape)
        return out

    def sample(self, inputs, states=None, max_len=20):
        """ accepts pre-processed image tensor (inputs) and
        returns predicted sentence (list of tensor ids of length max_len) """
        output_sentence = []
        for i in range(max_len):
            lstm_outputs, states = self.lstm(inputs, states)
            lstm_outputs = lstm_outputs.squeeze(1)
            out = self.linear(lstm_outputs)
            last_pick = out.max(1)[1]
            output_sentence.append(last_pick.item())
            inputs = self.embedding_layer(last_pick).unsqueeze(1)

        return output_sentence
