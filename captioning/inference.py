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

from config import Config
from data_handler.data_loader import get_data_loader
from utils import imshow, clean_sentence


def get_predict(images, encoder, decoder, vocab, captions=None):
    # image = image.to(device)
    # print(images.shape)
    for i in range(images.shape[0]):
        image = images[i].unsqueeze(0)
        print(image.shape)
        features = encoder(image).unsqueeze(1)
        output = decoder.sample(features)
        for index, s in enumerate(output):
            sentence = clean_sentence(s, vocab)
            print("Predicted Caption {}: ".format(index) + str(sentence))
        if captions is not None:
            print("Original Caption: " + str(captions[i]))
        imshow(image[0])
