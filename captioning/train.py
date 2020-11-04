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

import sys
from collections import Counter
from captioning_config import CaptioningConfig as Config
import numpy as np
import torch.utils.data as data
import torch
import math
import os

# import wandb
#
# wandb.init(project="test_ImageQuery")


def validate(val_loader, encoder, decoder, criterion, device):
    vocab_size = len(val_loader.dataset.vocab)
    with torch.no_grad():
        # set the evaluation mode
        encoder.eval()
        decoder.eval()
        val_indices = val_loader.dataset.get_train_indices()

        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        val_sampler = data.sampler.SubsetRandomSampler(indices=val_indices)
        val_loader.batch_sampler.sampler = val_sampler
        # get the validation images and captions
        val_images, val_captions = next(iter(val_loader))
        val_images = val_images.to(device)
        val_captions = val_captions.to(device)

        # define the captions
        # captions_target = val_captions[:, 1:]#.to(device)
        # captions_train = val_captions[:, : val_captions.shape[1] - 1]#.to(device)

        # Move batch of images and captions to GPU if CUDA is available.
        # val_images = val_images.to(device)

        # Pass the inputs through the CNN-RNN model.
        features = encoder(val_images)
        outputs = decoder(features, val_captions)

        # Calculate the batch loss.
        val_loss = criterion(outputs.view(-1, vocab_size), val_captions.view(-1))
        return val_loss


def train(encoder, decoder, optimizer, criterion, train_loader, val_loader, device):
    # f = open(Config.log_file, "w")
    losses = list()
    val_losses = list()
    vocab_size = len(train_loader.dataset.vocab)
    exp_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=Config.scheduler_gamma
    )

    total_step = math.ceil(
        len(train_loader.dataset) / train_loader.batch_sampler.batch_size
    )
    # set decoder and encoder into train mode
    # wandb.watch(encoder)
    # wandb.watch(decoder)
    encoder.train()
    decoder.train()
    for epoch in Config.epoch_range:

        for i_step in range(1, total_step + 1):

            # zero the gradients
            decoder.zero_grad()
            encoder.zero_grad()

            # Randomly sample a caption length, and sample indices with that length.
            indices = train_loader.dataset.get_train_indices()

            # Create and assign a batch sampler to retrieve a batch with the sampled indices.
            new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
            train_loader.batch_sampler.sampler = new_sampler

            # Obtain the batch.
            images, captions = next(iter(train_loader))
            images = images.to(device)
            captions = captions.to(device)

            # make the captions for targets and teacher forcer
            # captions_train = captions[:, :-1].to(device)
            # captions_target = captions.to(device)
            # print(captions_train.shape)
            # print(captions_target.shape)

            # Pass the inputs through the CNN-RNN model.
            features = encoder(images)
            # print(features.shape)
            outputs = decoder(features, captions)
            # print(outputs.view(-1, vocab_size).shape, captions.contiguous().view(-1).shape)
            # return
            # Calculate the batch loss
            loss = criterion(
                # outputs.view(-1, vocab_size), captions_target.contiguous().view(-1)
                outputs.view(-1, vocab_size),
                captions.contiguous().view(-1),
            )

            # Backward pass
            loss.backward()

            # Update the parameters in the optimizer
            optimizer.step()

            # - - - Validate - - -
            # turn the evaluation mode on
            val_loss = validate(val_loader, encoder, decoder, criterion, device)
            # encoder.train()
            # decoder.train()

            # append the validation loss and training loss
            val_losses.append(val_loss.item())
            losses.append(loss.item())

            # save the losses
            np.save("losses", np.array(losses))
            np.save("val_losses", np.array(val_losses))

            # Get training statistics.
            stats = "Epoch [%d/%d], Step [%d/%d], Train Loss: %.4f, Val Loss: %.4f" % (
                epoch,
                Config.epoch_range[-1],
                i_step,
                total_step,
                loss.item(),
                val_loss.item(),
            )

            if i_step % Config.print_every == 0:
                print(stats)
                # wandb.log({"train_loss": loss.item(), "val_loss": val_loss.item()})
                # sys.stdout.flush()
                # f.write(stats + "\n")
                # f.flush()
        # Save the weights.
        if epoch % Config.save_every == 0:
            print("\nSaving the model")
            torch.save(
                decoder.state_dict(), os.path.join("./models", "decoder-%d.pth" % epoch)
            )
            torch.save(
                encoder.state_dict(), os.path.join("./models", "encoder-%d.pth" % epoch)
            )
        exp_lr_scheduler.step()
    # f.close()
