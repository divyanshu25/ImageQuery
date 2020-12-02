#  ================================================================
#  Copyright 2020 Image Query Team
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==================================================================

from captioning.captioning_config import Config
import matplotlib.pyplot as plt
import numpy as np
import nltk
import torch
import math
from collections import Counter
from torchtext.data.metrics import bleu_score, _compute_ngram_counter


def imshow(img, txt=None):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.text(0, 0, s=txt, bbox=dict(facecolor="red", alpha=0.5))
    plt.show()


def clean_sentence(output, vocab, bert=None, use_bert=False):
    # output = output.numpy()
    words_sequence = []
    for i in output:
        if use_bert:
            words_sequence.append(bert.get_tokenizer().convert_ids_to_tokens(i))
            if i == 102:
                break
        else:
            words_sequence.append(vocab.idx2word[i])
            if i == 1:
                break

    words_sequence = words_sequence[1:-1]
    sentence = " ".join(words_sequence)
    sentence = sentence.capitalize()

    return sentence


def convert_captions(images, target, vocab, config, bert=None):
    # images, target = input
    all_captions = None
    caption_lengths = None

    if len(target) > 0:
        if not len(target) == config.batch_size:
            target = target[0]
        all_captions = []
        caption_lengths = []
        for c in target:
            caption = []
            if not config.enable_bert:
                tokens = nltk.tokenize.word_tokenize(str(c).lower())
                caption.append(vocab(vocab.start_word))
                caption.extend([vocab(token) for token in tokens])
                caption.append(vocab(vocab.end_word))
                cap_length = len(caption)
                if cap_length < config.max_length:
                    for i in range(config.max_length - len(caption)):
                        caption.append(vocab(vocab.pad_word))
                else:
                    caption = caption[0 : config.max_length - 1]
                    caption.append(vocab(vocab.end_word))
                    cap_length = len(caption)
                caption_lengths.append(cap_length)
            else:
                tokenizer = bert.get_tokenizer()
                tokens = tokenizer.tokenize(str(c).lower())
                caption.append(tokenizer.cls_token_id)
                caption.extend(
                    [tokenizer.convert_tokens_to_ids(token) for token in tokens]
                )
                caption.append(tokenizer.sep_token_id)
                cap_length = len(caption)
                if cap_length < config.max_length:
                    for i in range(config.max_length - len(caption)):
                        caption.append(tokenizer.sep_token_id)
                else:
                    caption = caption[0 : config.max_length - 1]
                    caption.append(tokenizer.sep_token_id)
                    cap_length = len(caption)
                caption_lengths.append(cap_length)
            all_captions.append(caption)
        # caption = caption[:1]
        all_captions = torch.Tensor(all_captions).long()
        caption_lengths = torch.Tensor(caption_lengths).unsqueeze(1).long()
    return images, all_captions, caption_lengths


def get_term_weights(references_corpus):
    term_count = Counter()
    for ref in references_corpus:
        count = _compute_ngram_counter(ref, 1)
        term_count = term_count + count
    total_count = sum(term_count.values())
    term_weights = {}
    for k in term_count:
        term_weights[k] = 1 - (term_count[k] * 2 / total_count)

    return term_weights


def custom_bleu(candidate_corpus, references_corpus, max_n=4, weights=[0.25]*4, term_weights={}):
    assert max_n == len(weights), 'Length of the "weights" list has be equal to max_n'
    assert len(candidate_corpus) == len(references_corpus), \
        'The length of candidate and reference corpus should be the same'

    clipped_counts = torch.zeros(max_n)
    total_counts = torch.zeros(max_n)
    weights = torch.tensor(weights)

    candidate_len = 0.0
    refs_len = 0.0

    for (candidate, refs) in zip(candidate_corpus, references_corpus):
        candidate_len += len(candidate)

        # Get the length of the reference that's closest in length to the candidate
        refs_len_list = [float(len(ref)) for ref in refs]
        refs_len += min(refs_len_list, key=lambda x: abs(len(candidate) - x))

        reference_counters = _compute_ngram_counter(refs[0], max_n)
        for ref in refs[1:]:
            reference_counters = reference_counters | _compute_ngram_counter(ref, max_n)

        candidate_counter = _compute_ngram_counter(candidate, max_n)

        clipped_counter = candidate_counter & reference_counters

        for ngram in clipped_counter:
            score = clipped_counter[ngram]
            if ngram in term_weights:
                score = clipped_counter[ngram] * term_weights[ngram]
            clipped_counts[len(ngram) - 1] += score

        for ngram in candidate_counter:  # TODO: no need to loop through the whole counter
            total_counts[len(ngram) - 1] += candidate_counter[ngram]

    if min(clipped_counts) == 0:
        return 0.0
    else:
        pn = clipped_counts / total_counts
        log_pn = weights * torch.log(pn)
        score = torch.exp(sum(log_pn))

        bp = math.exp(min(1 - refs_len / candidate_len, 0))

        return bp * score.item()
