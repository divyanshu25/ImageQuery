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

from captioning_config import Config as Config
import matplotlib.pyplot as plt
import numpy as np
import nltk
import torch


def imshow(img, txt=None):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.text(0, 0, s=txt, bbox=dict(facecolor="red", alpha=0.5))
    plt.show()


def display_image(images, labels, data_loader):
    # for l in labels:
    # print(l)
    for i in range(Config.batch_size):
        desc = clean_sentence(labels[i], data_loader)
        imshow(images[i], txt=desc)

    # show image
    # imshow(torchvision.utils.make_grid(images))


def clean_sentence(output, vocab):
    # output = output.numpy()
    words_sequence = []
    for i in output:
        words_sequence.append(vocab.idx2word[i])

    words_sequence = words_sequence[1:-1]
    sentence = " ".join(words_sequence)
    sentence = sentence.capitalize()

    return sentence


def convert_captions(input, vocab, config):
    images, target = input
    all_captions = None

    if len(target) > 0:
        if not len(target) == config.batch_size:
            target = target[0]
        all_captions = []
        for c in target:
            caption = []
            tokens = nltk.tokenize.word_tokenize(str(c).lower())
            caption.append(vocab(vocab.start_word))
            caption.extend([vocab(token) for token in tokens])
            caption.append(vocab(vocab.end_word))
            for i in range(config.max_length - len(caption)):
                caption.append(vocab(vocab.end_word))
            caption = caption[0 : config.max_length - 1]
            caption.append(vocab(vocab.end_word))
            all_captions.append(caption)
        # caption = caption[:1]
        all_captions = torch.Tensor(all_captions).long()
    return images, all_captions
