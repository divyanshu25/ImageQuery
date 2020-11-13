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
from captioning_config import Config

config = Config()


class EncoderAttn(nn.Module):
    def __init__(self, embed_dim):
        super(EncoderAttn, self).__init__()
        self.enc_image_size = config.encoderattn_encodedimgsize

        resnet = torchvision.models.resnet101(
            pretrained=True
        )  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (self.enc_image_size, self.enc_image_size)
        )

        self.fine_tune(config.train_encoder)

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(
            out
        )  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(
            0, 2, 3, 1
        )  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

    def get_learning_parameters(self):
        params = list(self.adaptive_pool.parameters())
        p = []
        if config.train_encoder:
            for c in list(self.resnet.children())[5:]:
                p = p + list(c.parameters())
        return params + p
