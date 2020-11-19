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

from captioning.utils import imshow, clean_sentence
from captioning.captioning_config import Config
import torch
import torch.nn.functional as F

config = Config()


def beam_search(encoder, decoder, image):
    beam_size = config.beam_size
    max_len = config.max_length
    encoder_out = encoder(image)
    encoder_out, state = decoder.init_search(encoder_out)
    sequences = [
        [0.0, torch.LongTensor([[0]]), [0], state]
    ]  # [Value, curr_word, output_sentence, states]
    if config.arch_name == "vanilla":
        sequences = [[0.0, torch.LongTensor([[0]]), [], state]]
    finished_beams = []
    best_so_far = 0.0

    for i in range(max_len):
        expanded_beams = []
        for index, s in enumerate(sequences):
            scores, state = decoder.predict_next(encoder_out, s[1], s[3])
            out = F.log_softmax(scores, dim=1)
            topk_picks = torch.topk(out, beam_size, dim=1)  #
            topk_picks_values = topk_picks[0].squeeze()
            topk_picks_indices = topk_picks[1].squeeze()
            for ix, val in zip(topk_picks_indices, topk_picks_values):
                current_beam = []
                current_beam.extend(
                    [
                        s[0] + val.item(),
                        torch.LongTensor([ix]),
                        s[2] + [ix.item()],
                        state,
                    ]
                )
                if ix.item() == 1:
                    finished_beams.append(current_beam)
                    if best_so_far < current_beam[0]:
                        best_so_far = current_beam[0]
                else:
                    expanded_beams.append(current_beam)

        ordered = sorted(expanded_beams, key=lambda tup: tup[0])[::-1]
        sequences = ordered[:beam_size]

    sequences.extend(finished_beams)
    ordered = sorted(sequences, key=lambda tup: tup[0])[::-1]
    output_sentences = []
    for beam in ordered[:beam_size]:
        output_sentences.append(beam[2])
    return output_sentences


def get_predict(images, encoder, decoder, vocab, captions=None, bert=None):
    for i in range(images.shape[0]):
        image = images[i].unsqueeze(0)
        output = beam_search(encoder, decoder, image)
        for index, s in enumerate(output):
            sentence = clean_sentence(s, vocab, bert=bert, use_bert=config.enable_bert)
            print("Predicted Caption {}: ".format(index) + str(sentence))
        if captions is not None:
            print(
                "Original Caption: " + clean_sentence(captions[i].cpu().numpy().tolist(), vocab, bert=bert, use_bert=config.enable_bert)
            )
        imshow(image[0])
