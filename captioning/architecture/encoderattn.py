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
from torch import nn
import torchvision
from captioning.captioning_config import Config
from torch.nn import AdaptiveAvgPool2d as AAP
from torchvision.models import resnet101  # , resnet50

config = Config()


class EncoderAttn(nn.Module):
    def __init__(self, embed_dim):
        super(EncoderAttn, self).__init__()
        self.enc_image_size = config.encoded_img_size

        self.pool = AAP((self.enc_image_size, self.enc_image_size))

        # resnet = resnet50(
        #     pretrained=True
        # )
        resnet = resnet101(pretrained=True)
        self.resnet = nn.Sequential(*(list(resnet.children())[:-2]))
        self.train_parameters(config.train_encoder)

    def forward(self, images):
        out = self.pool(self.resnet(images)).permute(0, 2, 3, 1)
        return out

    def train_parameters(self, is_trainable=True):
        for p in self.resnet.parameters():
            p.requires_grad = False
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = is_trainable

    def get_learning_parameters(self):
        params = list(self.pool.parameters())
        p = []
        if config.train_encoder:
            for c in list(self.resnet.children())[5:]:
                p = p + list(c.parameters())
        return params + p
