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
from collections import Counter

from captioning.config import Config
from captioning.data_handler.data_loader import get_data_loader
from captioning.utils import display_image
from captioning.data_handler.utils import parse_flickr
import numpy as np
import torch.utils.data as data


def execute():
    # Step1: Load Data
    flickr_ann_dict = parse_flickr(Config.annotations_file)
    train_loader = get_data_loader(Config, flickr_ann_dict, mode="train")
    # print("Total number of tokens in vocabulary:", len(train_loader.dataset.vocab))
    # print("Total number of tokens in vocabulary:", len(train_loader.dataset.vocab))
    # print(train_loader.dataset.vocab("ieowoqjf"))
    # counter = Counter(train_loader.dataset.caption_lengths)
    # lengths = sorted(counter.items(), key=lambda pair: pair[1], reverse=True)
    # for value, count in lengths:
    #     print("value: %2d --- count: %5d" % (value, count))

    indices = train_loader.dataset.get_train_indices()
    print("sampled indices:", indices)
    new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
    train_loader.batch_sampler.sampler = new_sampler

    # Obtain the batch.
    images, captions = next(iter(train_loader))

    print('images.shape:', images.shape)
    print('captions.shape:', captions.shape)
    # display_image(train_loader)

    # Step2: Define and Initialize Neural Net/ Model Class/ Hypothesis(H).
    # encoder = EncoderCNN()
    # decoder = DecoderRNN()

    # Step3: Define Loss Function and optimizer
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Step4: Train the network.
    # train(net, optimizer, criterion, train_loader)
    #
    # # Step5: Save Model
    # PATH = "./model/cifar_net.pth"
    # # torch.save(net.state_dict(), PATH)
    #
    # # Step6: Test the net on Test Data
    # net = Net()
    # net.load_state_dict(torch.load(PATH))
    # run_test(test_loader, net, classes)


# def train():
#     losses = list()
#     val_losses = list()
#
#     for epoch in range(1, 10 + 1):
#
#         for i_step in range(1, total_step + 1):
#
#             # zero the gradients
#             decoder.zero_grad()
#             encoder.zero_grad()
#
#             # set decoder and encoder into train mode
#             encoder.train()
#             decoder.train()
#
#             # Randomly sample a caption length, and sample indices with that length.
#             indices = train_data_loader.dataset.get_train_indices()
#
#             # Create and assign a batch sampler to retrieve a batch with the sampled indices.
#             new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
#             train_data_loader.batch_sampler.sampler = new_sampler
#
#             # Obtain the batch.
#             images, captions = next(iter(train_data_loader))
#
#             # make the captions for targets and teacher forcer
#             captions_target = captions[:, 1:].to(device)
#             captions_train = captions[:, : captions.shape[1] - 1].to(device)
#
#             # Move batch of images and captions to GPU if CUDA is available.
#             images = images.to(device)
#
#             # Pass the inputs through the CNN-RNN model.
#             features = encoder(images)
#             outputs = decoder(features, captions_train)
#
#             # Calculate the batch loss
#             loss = criterion(
#                 outputs.view(-1, vocab_size), captions_target.contiguous().view(-1)
#             )
#
#             # Backward pass
#             loss.backward()
#
#             # Update the parameters in the optimizer
#             optimizer.step()
#
#             # - - - Validate - - -
#             # turn the evaluation mode on
#             with torch.no_grad():
#
#                 # set the evaluation mode
#                 encoder.eval()
#                 decoder.eval()
#
#                 # get the validation images and captions
#                 val_images, val_captions = next(iter(val_data_loader))
#
#                 # define the captions
#                 captions_target = val_captions[:, 1:].to(device)
#                 captions_train = val_captions[:, : val_captions.shape[1] - 1].to(device)
#
#                 # Move batch of images and captions to GPU if CUDA is available.
#                 val_images = val_images.to(device)
#
#                 # Pass the inputs through the CNN-RNN model.
#                 features = encoder(val_images)
#                 outputs = decoder(features, captions_train)
#
#                 # Calculate the batch loss.
#                 val_loss = criterion(
#                     outputs.view(-1, vocab_size), captions_target.contiguous().view(-1)
#                 )
#
#             # append the validation loss and training loss
#             val_losses.append(val_loss.item())
#             losses.append(loss.item())
#
#             # save the losses
#             np.save("losses", np.array(losses))
#             np.save("val_losses", np.array(val_losses))
#
#             # Get training statistics.
#             stats = "Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Val Loss: %.4f" % (
#                 epoch,
#                 num_epochs,
#                 i_step,
#                 total_step,
#                 loss.item(),
#                 val_loss.item(),
#             )
#
#             # Print training statistics (on same line).
#             print("\r" + stats, end="")
#             sys.stdout.flush()
#
#         # Save the weights.
#         if epoch % save_every == 0:
#             print("\nSaving the model")
#             torch.save(
#                 decoder.state_dict(), os.path.join("./models", "decoder-%d.pth" % epoch)
#             )
#             torch.save(
#                 encoder.state_dict(), os.path.join("./models", "encoder-%d.pth" % epoch)
#             )


if __name__ == "__main__":
    execute()
