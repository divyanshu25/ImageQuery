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

from .encoder import EncoderCNN
from .decoder import DecoderRNN
from .encoderattn import EncoderAttn
from .decoderattn import DecoderAttn

from captioning.captioning_config import Config as Config

config = Config()


def get_encoder_decoder(embed_size, hidden_size, vocab_size, bert=None):
    if config.arch_name == "vanilla":
        return (EncoderCNN(embed_size), DecoderRNN(embed_size, hidden_size, vocab_size))
    elif config.arch_name == "attention":
        return (
            EncoderAttn(embed_size),
            DecoderAttn(embed_size, hidden_size, vocab_size, bert=bert),
        )
