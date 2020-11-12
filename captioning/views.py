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
import math

from flask import make_response
from flask_apispec import marshal_with, doc, use_kwargs
from flask_apispec import MethodResource as Resource

from captioning.architecture.decoder import DecoderRNN
from captioning.architecture.encoder import EncoderCNN
from captioning.schema import PopulateImageSchema
from captioning.captioning_config import Config
from captioning.data_handler.data_loader import get_data_loader, get_vocabulary
from captioning.main import get_device
from models import ImageCaptions, db
import os
import torch

from captioning.utils import convert_captions, clean_sentence


@doc(
    summary="Load image captions",
    description="API end point to load image data from disk to the Database.",
    tags=["Captioning"],
)
class PopulateFlickrData(Resource):
    @marshal_with(PopulateImageSchema, code=200)
    def get(self, **kwargs):
        config = Config()
        data_loader = get_data_loader(config, mode="test", type=config.dataset_type)
        vocab = get_vocabulary(config, type=config.dataset_type)
        vocab_size = len(vocab)

        device = get_device()

        encoder = EncoderCNN(config.embed_size)
        decoder = DecoderRNN(config.embed_size, config.hidden_size, vocab_size)
        if device:
            encoder = encoder.cuda()
            decoder = decoder.cuda()

        encoder.eval()
        decoder.eval()
        if not torch.cuda.is_available():
            encoder.load_state_dict(
                torch.load(config.encoder_file, map_location=torch.device("cpu"))
            )
            decoder.load_state_dict(
                torch.load(config.decoder_file, map_location=torch.device("cpu"))
            )
        else:
            encoder.load_state_dict(torch.load(config.encoder_file))
            decoder.load_state_dict(torch.load(config.decoder_file))

        total_step = math.ceil(
            len(data_loader.dataset) / data_loader.batch_sampler.batch_size
        )
        for i in range(total_step):
            print("step: {}",format(i))
            images, captions, image_ids = next(iter(data_loader))
            images, captions = convert_captions(images, captions, vocab, config)
            if device:
                images = images.cuda()

            for k in range(images.shape[0]):
                image = images[k].unsqueeze(0)
                features = encoder(image).unsqueeze(1)
                output = decoder.sample(features)
                for index, s in enumerate(output):
                    sentence = clean_sentence(s, vocab)
                    print("Predicted Caption {}: ".format(index) + str(sentence))
                if captions is not None:
                    print("Original Caption: " + clean_sentence(captions[k].cpu().numpy(), vocab))


        # print(os.getcwd())
        # data_file = Config.dummy_annotations_file
        # try:
        #     with open(data_file, "r") as f:
        #         for l in f.readlines():
        #             tokens = l.split("#")
        #             image_id = tokens[0]
        #             caption = tokens[1][2:]
        #             encoded_caption = bert_encoder.get_bert_enoding(caption)
        #             caption_index = int(tokens[1][0])
        #             captions_obj = ImageCaptions(
        #                 image_path=image_id,
        #                 caption_index=caption_index,
        #                 caption=caption,
        #                 encoded_caption=encoded_caption,
        #             )
        #             db.session.add(captions_obj)
        #             db.session.commit()
        #         return make_response(dict(status="Data Upload Success"), 200)
        # except Exception as e:
        #     print(e)
        #     return make_response(dict(status="Data Upload Failed"), 500)
