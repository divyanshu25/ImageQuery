import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


def imshow(img, txt=None):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.text(0, 0, s=txt, bbox=dict(facecolor="red", alpha=0.5))
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
