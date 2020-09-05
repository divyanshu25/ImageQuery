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


import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.googlenet(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


# class EncoderCNN(nn.Module):
#     def __init__(self, embed_size=1024):
#         super(EncoderCNN, self).__init__()
#
#         # get the pretrained densenet model
#         self.densenet = models.densenet121(pretrained=True)
#
#         # replace the classifier with a fully connected embedding layer
#         self.densenet.classifier = nn.Linear(in_features=1024, out_features=1024)
#
#         # add another fully connected layer
#         self.embed = nn.Linear(in_features=1024, out_features=embed_size)
#
#         # dropout layer
#         self.dropout = nn.Dropout(p=0.5)
#
#         # activation layers
#         self.prelu = nn.PReLU()
#
#     def forward(self, images):
#
#         # get the embeddings from the densenet
#         densenet_outputs = self.dropout(self.prelu(self.densenet(images)))
#
#         # pass through the fully connected
#         embeddings = self.embed(densenet_outputs)
#
#         return embeddings
