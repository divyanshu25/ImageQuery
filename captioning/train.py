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

from captioning.captioning_config import Config
import torch
import math
import os
from captioning.utils import convert_captions, clean_sentence
import wandb
from torch.nn.utils.rnn import pack_padded_sequence


config = Config()
if config.enable_wandb:
    wandb.init(project="test_ImageQuery")


def validate(val_loader, encoder, decoder, criterion, device, vocab):
    vocab_size = len(vocab)
    with torch.no_grad():
        # set the evaluation mode
        encoder.eval()
        decoder.eval()
<<<<<<< HEAD

=======
>>>>>>> Beam Search for attention Model
        val_images, val_captions, _ = next(iter(val_loader))
        val_images, val_captions, caption_lengths = convert_captions(
            val_images, val_captions, vocab, config
        )

        if device:
            val_images = val_images.cuda()
            val_captions = val_captions.cuda()
            caption_lengths = caption_lengths.cuda()

        # Pass the inputs through the CNN-RNN model.
        features = encoder(val_images)
        outputs, caps_sorted, decode_lengths, alphas = decoder(
            features, val_captions, caption_lengths
        )
        targets = caps_sorted[:, 1:]
        scores = pack_padded_sequence(outputs, decode_lengths, batch_first=True)
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)
        if device:
            scores = scores.cuda()
            targets = targets.cuda()

        val_loss = criterion(scores.data, targets.data)
        val_loss += 1 * ((1.0 - alphas.sum(dim=1)) ** 2).mean()
        encoder.train()
        decoder.train()
        return val_loss


def train(
    encoder, decoder, optimizer, criterion, train_loader, val_loader, device, vocab
):

    vocab_size = len(vocab)
    exp_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=config.scheduler_gamma
    )

    total_step = math.ceil(
        len(train_loader.dataset) / train_loader.batch_sampler.batch_size
    )
    # set decoder and encoder into train mode
    if config.enable_wandb:
        wandb.watch(encoder)
        wandb.watch(decoder)

    encoder.train()
    decoder.train()
    for epoch in config.epoch_range:

        for i_step in range(1, total_step + 1):
            # zero the gradients
            decoder.zero_grad()
            encoder.zero_grad()
            # Obtain the batch.
            images, captions, _ = next(iter(train_loader))
            images, captions, caption_lengths = convert_captions(images, captions, vocab, config)
            if device:
                images = images.cuda()
                captions = captions.cuda()
                caption_lengths = caption_lengths.cuda()

            # Pass the inputs through the CNN-RNN model.
            features = encoder(images)
            # print(features.shape)
            # Checkpoint[2]
            outputs, caps_sorted, decode_lengths, alphas = decoder(
                features, captions, caption_lengths
            )
            # print(f"Output:{outputs.size()}, cap_sorted:{caps_sorted}, decode_len:{decode_lengths},"
            #       f"alphas: {alphas.size()}, sortind:{sort_ind}")
            # Checkpoint[3]
            targets = caps_sorted[:, 1:]
            # print(f"scores:{outputs}, targets:{targets}")
            scores = pack_padded_sequence(outputs, decode_lengths, batch_first=True)
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)
            # print(f"scores:{scores.data},\n "
            #       f"targets:{targets.data}")
            # return
            # Checkpoint[4]
            # Calculate the batch loss
            if device:
                scores = scores.cuda()
                targets = targets.cuda()
            loss = criterion(scores.data, targets.data)
            loss += 1 * ((1.0 - alphas.sum(dim=1)) ** 2).mean()

            if config.verbose and i_step == total_step:
                for batch in range(min(config.batch_size, 10)):
                    curr_pred_vec = outputs[batch, :, :]
                    predicted_caption = torch.max(curr_pred_vec, dim=1)
                    print(
                        "Predicted_caption_Indices: ",
                        clean_sentence(predicted_caption.indices.cpu().numpy(), vocab),
                    )
                    print(
                        "Original Caption Indices: ",
                        clean_sentence(captions[batch].cpu().numpy(), vocab),
                    )
                print("=================================")

            # Backward pass
            loss.backward()
            # Update the parameters in the optimizer
            optimizer.step()
            # - - - Validate - - -

            if i_step % config.print_every == 0:
                val_loss = validate(val_loader, encoder, decoder, criterion, device, vocab)
                # Get training statistics.
                stats = "Epoch [%d/%d], Step [%d/%d], Train Loss: %.4f, Val Loss: %.4f" % (
                    epoch,
                    config.epoch_range[-1],
                    i_step,
                    total_step,
                    loss.item(),
                    val_loss.item(),
                )
                print(stats)
                if config.enable_wandb:
                    wandb.log({"train_loss": loss.item(), "val_loss": val_loss.item()})
        # Save the weights.
        if epoch % config.save_every == 0:
            print("\nSaving the model")
            torch.save(
                decoder.state_dict(),
                os.path.join(
                    config.models_dir, "{}-{}.pth".format(config.decoder_prefix, epoch)
                ),
            )
            torch.save(
                encoder.state_dict(),
                os.path.join(
                    config.models_dir, "{}-{}.pth".format(config.encoder_prefix, epoch)
                ),
            )
        exp_lr_scheduler.step()
