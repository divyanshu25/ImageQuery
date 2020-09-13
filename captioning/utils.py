from captioning.config import Config
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


def imshow(img, txt=None):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.text(0, 0, s=txt, bbox=dict(facecolor="red", alpha=0.5))
    plt.show()


def display_image(images, labels, data_loader):
    # for l in labels:
    #     print(l)
    for i in range(Config.batch_size):
        print(clean_sentence(labels[i], data_loader))
        imshow(images[i], labels[i])

    # show image
    # imshow(torchvision.utils.make_grid(images))


def clean_sentence(output, data_loader):
    words_sequence = []

    for i in output:
        if (i == 1):
            continue
        words_sequence.append(data_loader.dataset.vocab.idx2word[i])

    words_sequence = words_sequence[1:-1]
    sentence = ' '.join(words_sequence)
    sentence = sentence.capitalize()

    return sentence