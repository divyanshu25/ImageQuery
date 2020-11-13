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

import nltk
from flask import make_response, request
from flask_apispec import marshal_with, doc, use_kwargs
from flask_apispec import MethodResource as Resource

from captioning.architecture.decoder import DecoderRNN
from captioning.architecture.encoder import EncoderCNN
from captioning.schema import PopulateImageSchema, BleuScoreSchema, PopulateSearchSchema
from captioning.captioning_config import Config
from captioning.data_handler.data_loader import get_data_loader, get_vocabulary
from captioning.main import get_device
from models import ImageCaptions, db
import os
import torch
import numpy as np
from torchtext.data.metrics import bleu_score
from captioning.utils import convert_captions, clean_sentence
from sklearn.metrics.pairwise import cosine_similarity


@doc(
    summary="Load image captions",
    description="API end point to load image data from disk to the Database.",
    tags=["Captioning"],
)
class PopulateFlickrData(Resource):
    @marshal_with(PopulateImageSchema, code=200)
    def get(self, set):
        if set not in ["test", "val", "train"]:
            return make_response(dict(status="Invalid Set"), 500)
        config = Config()
        data_loader = get_data_loader(config, mode=set, type=config.dataset_type)
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
        print("Total Steps: {}".format(total_step))
        try:
            for i in range(total_step):
                print("step: {}".format(i))
                images, captions, image_ids = next(iter(data_loader))
                images, captions = convert_captions(images, captions, vocab, config)
                if device:
                    images = images.cuda()

                for k in range(images.shape[0]):
                    image = images[k].unsqueeze(0)
                    features = encoder(image).unsqueeze(1)
                    output = decoder.sample(features)
                    image_id = image_ids[k].split("#")[0]
                    if not db.session.query(
                        db.session.query(ImageCaptions)
                        .filter_by(image_path=image_id)
                        .exists()
                    ).scalar():
                        for index, beam in enumerate(output):
                            sentence = clean_sentence(beam, vocab)
                            caption_index = str(index + 5)
                            captions_obj = ImageCaptions(
                                image_path=image_id,
                                caption_index=caption_index,
                                set="flickr_{}".format(set),
                                caption=sentence,
                                # encoded_caption=str(beam),
                            )
                            print(
                                "Inserting Prediction {}, {}".format(
                                    image_id, caption_index
                                )
                            )
                            db.session.add(captions_obj)
                            db.session.commit()
                    else:
                        print("Already Exist: {}".format(image_id))

                    if captions is not None:
                        sentence = clean_sentence(captions[k].cpu().numpy(), vocab)
                        caption_index = image_ids[k].split("#")[1]

                        if not db.session.query(
                            db.session.query(ImageCaptions)
                            .filter_by(image_path=image_id, caption_index=caption_index)
                            .exists()
                        ).scalar():
                            encoded_caption = captions[k].cpu().numpy()
                            endindex = np.where(encoded_caption == 1)[0][0]
                            # encoded_caption = encoded_caption[: endindex + 1]
                            captions_obj = ImageCaptions(
                                image_path=image_id,
                                caption_index=caption_index,
                                set="flickr_{}".format(set),
                                caption=sentence,
                                # encoded_caption=str(encoded_caption.tolist()),
                            )
                            print(
                                "Inserting Prediction {}, {}".format(
                                    image_id, caption_index
                                )
                            )
                            db.session.add(captions_obj)
                            db.session.commit()
                        else:
                            print(
                                "Alredy Exist: {}, {}".format(image_id, caption_index)
                            )
            return make_response(dict(status="Data Upload Success"), 200)
        except Exception as e:
            print(e)
            return make_response(dict(status="Data Upload Failed"), 500)


@doc(
    summary="Compute Bleu Score", description="Compute Bleu Score", tags=["Captioning"]
)
class ComputeBleu(Resource):
    @marshal_with(BleuScoreSchema, code=200)
    def get(self, model_name, set):
        if model_name not in ["coco", "flickr"] or set not in ["test", "train", "val"]:
            return make_response(dict(status="Invalid set or model name"), 500)
        data = (
            db.session.query(ImageCaptions)
            .filter_by(set="{}_{}".format(model_name, set))
            .all()
        )
        predicted_captions = {}
        original_captions = {}
        for d in data:
            image_id = d.image_path
            caption_id = d.caption_index
            if caption_id < 5:
                if image_id not in original_captions:
                    original_captions[image_id] = []
                original_captions[image_id].append(
                    nltk.tokenize.word_tokenize(d.caption.lower())
                )
            else:
                if image_id not in predicted_captions:
                    predicted_captions[image_id] = []
                predicted_captions[image_id].append(
                    nltk.tokenize.word_tokenize(d.caption.lower())
                )
        reference_corpus = []  # original
        candidate_corpus = []  # predicted

        for k, v in predicted_captions.items():
            for caption in v:
                candidate_corpus.append(caption)
                reference_corpus.append(original_captions[k])
        # print(candidate_corpus)
        # print(reference_corpus)

        score = bleu_score(
            candidate_corpus, reference_corpus, max_n=2, weights=[0.25, 0.25]
        )
        return make_response(dict(bleu_Score=score), 200)


@doc(
    summary="Search Image given query",
    description="Search Image given query",
    tags=["Captioning"],
)
class SearchImage(Resource):
    @marshal_with(PopulateSearchSchema, code=200)
    def get(self, model_name, query):
        config = Config()

        if model_name not in ["coco", "flickr"]:
            return make_response(dict(status="Invalid model name"), 500)
        if model_name == "flickr":
            vocab = get_vocabulary(config, "flickr8k")
        else:
            vocab = get_vocabulary(config, "coco")
        decoder = DecoderRNN(config.embed_size, config.hidden_size, len(vocab))
        caption = self.get_encodings(vocab, query, config, decoder)

        data = (
            db.session.query(ImageCaptions)
            .filter_by(set="{}_{}".format(model_name, "test"))
            .all()
        )
        cosine_scores = {}
        for d in data:
            caption_index = d.caption_index
            if caption_index >= 5:
                cosine_score = self.get_cosine_similarity(
                    caption, self.get_encodings(vocab, d.caption, config, decoder)
                )
                if d.image_path not in cosine_scores:
                    cosine_scores[d.image_path] = 0
                cosine_scores[d.image_path] = max(
                    cosine_scores[d.image_path], cosine_score[0][0]
                )
        sorted_dict = {
            k: v
            for k, v in sorted(
                cosine_scores.items(), key=lambda item: item[1], reverse=True
            )
        }
        print(sorted_dict)
        list_images = []
        count = 0
        for k, v in sorted_dict.items():
            list_images.append(k)
            count += 1
            if count == 5:
                break
        return make_response(dict(image_ids=str(list_images)), 200)

    def get_cosine_similarity(self, a, b):
        return cosine_similarity(a.detach().numpy(), b.detach().numpy())

    def get_encodings(self, vocab, query, config, decoder):
        tokens = nltk.tokenize.word_tokenize(query)
        caption = [vocab(vocab.start_word)]
        caption.extend([vocab(token) for token in tokens])
        caption.extend([vocab(vocab.end_word)])
        for i in range(config.max_length - len(caption)):
            caption.append(vocab(vocab.end_word))
        caption = caption[0 : config.max_length - 1]
        caption.append(vocab(vocab.end_word))
        caption = torch.Tensor(caption).long().unsqueeze(0)
        encodings = decoder.embedding_layer(caption).squeeze(0)
        encodings = encodings.max(dim=1)
        return encodings.indices.unsqueeze(0)
