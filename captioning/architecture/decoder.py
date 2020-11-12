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
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)

        self.project_feature_layer = nn.Linear(embed_size, hidden_size)

        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.linear = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features, captions, caption_lengths):
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
        # input 1,300
        beam_size = config.beam_size
        sequences = [
            [1.0, inputs, states, []]
        ]  # [Value, inputs, states, output_sentence]
        finished_beams = []
        best_so_far = 0.0
        for i in range(max_len):
            expanded_beams = []
            for s in sequences:
                lstm_outputs, states = self.lstm(s[1], s[2])
                lstm_outputs = lstm_outputs.squeeze(1)  # 1,512
                out = self.linear(lstm_outputs)  # 1,3004
                out = self.softmax(out)
                topk_picks = torch.topk(out, beam_size, dim=1)
                topk_picks_values = topk_picks[0].squeeze()
                topk_picks_indices = topk_picks[1].squeeze()
                for ix, val in zip(topk_picks_indices, topk_picks_values):
                    current_beam = []
                    current_beam.extend(
                        [
                            s[0] * val.item(),
                            self.embedding_layer(ix).unsqueeze(0).unsqueeze(0),
                            states,
                            s[3] + [ix.item()],
                        ]
                    )
                    if ix.item() == 1:
                        finished_beams.append(current_beam)
                        if best_so_far < current_beam[0]:
                            best_so_far = current_beam[0]
                    else:
                        expanded_beams.append(current_beam)

            ordered = sorted(expanded_beams, key=lambda tup: tup[0])[::-1]
            # if ordered[0][0] < best_so_far:
            #     break
            sequences = ordered[:beam_size]
        sequences.extend(finished_beams)
        ordered = sorted(sequences, key=lambda tup: tup[0])[::-1]
        output_sentences = []
        for beam in ordered[:beam_size]:
            output_sentences.append(beam[3])
        return output_sentences


