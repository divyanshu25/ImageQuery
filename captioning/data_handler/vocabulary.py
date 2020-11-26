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

import nltk

nltk.download("punkt")
import pickle
import os.path
from collections import Counter


class Vocabulary(object):
    def __init__(
        self,
        vocab_threshold,
        vocab_file,
        # annotations_file,
        start_word="<start>",
        end_word="<end>",
        unk_word="<unk>",
        pad_word="<pad>",
        vocab_from_file=False,
    ):
        """Initialize the vocabulary.
        Args:
          vocab_threshold: Minimum word count threshold.
          vocab_file: File containing the vocabulary.
          start_word: Special word denoting sentence start.
          end_word: Special word denoting sentence end.
          unk_word: Special word denoting unknown words.
          annotations_file: Path for train annotation file.
          vocab_from_file: If False, create vocab from scratch & override any existing vocab_file
                           If True, load vocab from from existing vocab_file, if it exists
        """
        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.pad_word = pad_word
        # self.annotations_file = annotations_file
        self.vocab_from_file = vocab_from_file
        if os.path.exists(self.vocab_file) and self.vocab_from_file:
            with open(self.vocab_file, "rb") as f:
                vocab = pickle.load(f)
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
                self.frequency = vocab.frequency
            self.idx = len(self.word2idx)
            print(
                "Vocabulary successfully loaded from {} file!".format(self.vocab_file)
            )
        else:
            self.build_vocab()
            # with open(self.vocab_file, "wb") as f:
            #     pickle.dump(self, f)
            print("Vocabulary successfully build from annotations file!")

    def build_vocab(self):
        """Populate the dictionaries for converting tokens to integers (and vice-versa)."""
        self.init_vocab()
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self.add_word(self.pad_word)
        # self.add_captions()

    def init_vocab(self):
        """Initialize the dictionaries for converting tokens to integers (and vice-versa)."""
        self.word2idx = {}
        self.idx2word = {}
        self.frequency = Counter()
        self.idx = 0

    def add_word(self, word):
        """Add a token to the vocabulary."""
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word

            self.idx += 1

    def add_captions(self, ann_dict):
        """Loop over training captions and add all tokens to the vocabulary that meet or exceed the threshold."""
        # ann_dict = parse_flickr(self.annotations_file)

        ids = ann_dict.keys()
        for i, id in enumerate(ids):
            caption = str(ann_dict[id])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            self.frequency.update(tokens)

            if i % 1000 == 0:
                print("[%d/%d] Tokenizing captions..." % (i, len(ids)))
        words = [word for word, cnt in self.frequency.items() if cnt >= self.vocab_threshold]
        # print(words)

        for i, word in enumerate(words):
            self.add_word(word)

    def dump_vocab_in_file(self):
        with open(self.vocab_file, "wb") as f:
            pickle.dump(self, f)

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
