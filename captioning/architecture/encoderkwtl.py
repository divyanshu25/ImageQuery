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

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from captioning_config import Config

config = Config()

class EncoderKWTL(nn.Module):
    def __init__(self, embed_size):
        super(EncoderKWTL,self).__init__()
        resnet = models.resnet101(pretrained = True)
        all_modules = list(resnet.children())
        modules = all_modules[:-2]
        self.resnet = nn.Sequential(*modules)
        self.avgpool = nn.AvgPool2d(7)
        self.fine_tune(status = config.train_encoder)    # To fine-tune the CNN, self.fine_tune(status = True)

    def forward(self,images):
        """
        The forward propagation function
        input: resized image of shape (batch_size,3,224,224)
        """
        #Run the image through the ResNet
        encoded_image = self.resnet(images)         # (batch_size,2048,7,7)
        batch_size = encoded_image.shape[0]
        features = encoded_image.shape[1]
        num_pixels = encoded_image.shape[2] * encoded_image.shape[3]
        # Get the global features of the image
        global_features = self.avgpool(encoded_image).view(batch_size, -1)   # (batch_size, 2048)
        enc_image = encoded_image.permute(0, 2, 3, 1)  #  (batch_size,7,7,2048)
        enc_image = enc_image.view(batch_size,num_pixels,features)          # (batch_size,num_pixels,2048)
        return (enc_image, global_features)

    def fine_tune(self, status = False):
        if not status:
            for param in self.resnet.parameters():
                param.requires_grad = False
        else:
            for module in list(self.resnet.children())[7:]:    #1 layer only. len(list(resnet.children())) = 8
                for param in module.parameters():
                    param.requires_grad = True

    def get_learning_parameters(self):
        params = list(self.avgpool.parameters())
        if config.train_encoder:
            params = params + list(self.resnet[7].parameters()) #only the last layer's parmaeters
        return params
