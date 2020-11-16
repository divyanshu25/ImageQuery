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

class Attention(nn.Module):
    def __init__(self, encoder_size, decoder_size, attention_size):
        super(Attention, self).__init__()
        self.attn_features = nn.Linear(
            encoder_size, attention_size
        )
        self.attn_embeddings = nn.Linear(
            decoder_size, attention_size
        )
        self.attn_complete = nn.Linear(
            attention_size, 1
        )
        self.relu = nn.ReLU()

    def forward(self, encoder_out, decoder_out):
        attn = self.attn_features(encoder_out) + self.attn_embeddings(decoder_out).unsqueeze(1)
        alpha = F.softmax(self.attn_complete(self.relu(attn)).squeeze(2), dim=1)
        attn_weighted = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return attn_weighted, alpha
