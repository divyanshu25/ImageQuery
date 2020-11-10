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


import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from captioning_config import Config

config = Config()


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad_(config.train_encoder)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_size)
        self.resnet.fc.requires_grad_(True)

    def forward(self, images):
        features = self.resnet(images)
        # features = features.view(features.size(0), -1)
        # features = self.embed(features)
        return features
