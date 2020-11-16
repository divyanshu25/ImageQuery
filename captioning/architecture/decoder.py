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

# from captioning_config import Config as Config
from captioning.captioning_config import Config

config = Config()


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.project_feature_layer = nn.Linear(embed_size, hidden_size)

        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.linear = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features, captions):
        captions = captions[:, :-1]
        # print(captions.shape)
        embed = self.embedding(captions)
        # print(embed.shape)
        embed = torch.cat((features.unsqueeze(1), embed), dim=1)
        # print(embed.shape)
        lstm_outputs, _ = self.lstm(embed)
        # print(lstm_outputs.shape)
        out = self.linear(lstm_outputs)
        # print(out.shape)
        return out

    def init_search(self, inputs, states=None, max_len=20):
        """ accepts pre-processed image tensor (inputs) and
        returns predicted sentence (list of tensor ids of length max_len) """
        # input 1,300
        states = (None, 0)
        inputs = inputs.unsqueeze(1)
        return (inputs, states)

    def predict_next(self, encoder_out, current_word, state):
        lstm_states, iteration = state
        if iteration == 0:
            lstm_outputs, lstm_states = self.lstm(encoder_out, lstm_states)
        else:
            current_word = self.embedding(current_word).unsqueeze(0)

            lstm_outputs, lstm_states = self.lstm(current_word, lstm_states)

        lstm_outputs = lstm_outputs.squeeze(1)  # 1,512
        out = self.linear(lstm_outputs)  # 1,3004

        return out, (lstm_states, iteration + 1)
