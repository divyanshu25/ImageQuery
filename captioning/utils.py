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
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


def imshow(img, txt):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.text(0, 0, s= txt, bbox=dict(facecolor='red', alpha=0.5))
    plt.show()


def display_image(train_loader):
    data_iter = iter(train_loader)
    images, labels = data_iter.next()
    # for l in labels:
    #     print(l)
    for i in range(4):
        print(labels[i])
        imshow(images[i], labels[i])

    # show image
    # imshow(torchvision.utils.make_grid(images))


def parseAnnotations(ann_file):
    ann_dict = {}
    with open(ann_file, "r") as f:
        lines = f.read().splitlines()
        for l in lines:
            tokens = l.split("#")
            if tokens[0] in ann_dict:
                ann_dict[tokens[0]].append(tokens[1][2:])
            else:
                ann_dict[tokens[0]] = [tokens[1][2:]]
    return ann_dict

    # print(" ".join("%5s" % classes[labels[j]] for j in range(4)))
