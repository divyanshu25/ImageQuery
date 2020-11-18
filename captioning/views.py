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
import traceback
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from flask import make_response, request
from flask_apispec import marshal_with, doc, use_kwargs
from flask_apispec import MethodResource as Resource
from captioning.architecture.archpicker import get_encoder_decoder
from captioning.schema import PopulateImageSchema, BleuScoreSchema, PopulateSearchSchema
from captioning.captioning_config import Config
from captioning.data_handler.data_loader import get_data_loader, get_vocabulary
from captioning.main import get_device
from captioning.inference import beam_search
from models import ImageCaptions, db
import torch
from torchtext.data.metrics import bleu_score
from captioning.utils import convert_captions, clean_sentence
import sys

stopset = set(stopwords.words("english"))


@doc(
    summary="Load image captions",
    description="API end point to load image data from disk to the Database.",
    tags=["Captioning"],
)
class PopulateData(Resource):
    @marshal_with(PopulateImageSchema, code=200)
    def get(self, model_name, set):
        if model_name not in [
            "coco",
            "flickr",
            "flickr_attn",
            "coco_attn",
        ] or set not in ["test", "train", "val"]:
            return make_response(dict(status="Invalid set or model name"), 500)
        config = Config()
        data_loader = get_data_loader(config, mode=set, type=config.dataset_type)
        vocab = get_vocabulary(config, type=config.dataset_type)
        vocab_size = len(vocab)

        device = get_device()

        encoder, decoder = get_encoder_decoder(
            config.embed_size, config.hidden_size, vocab_size
        )
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
                images, encoded_captions, caption_lengths = convert_captions(
                    images, captions, vocab, config
                )
                if device:
                    images = images.cuda()

                if "flickr" in model_name:
                    num_captions = 1
                    if captions is not None:
                        captions = [captions]
                else:
                    num_captions = 5

                for k in range(images.shape[0]):
                    image = images[k].unsqueeze(0)
                    output = beam_search(encoder, decoder, image)
                    # image_id = image_ids[k].split("#")[0]
                    if "flickr" in model_name:
                        image_id = image_ids[k].split("#")[0]
                    else:
                        image_id = image_ids[k].item()
                    if not db.session.query(
                        db.session.query(ImageCaptions)
                        .filter_by(
                            image_path=image_id, set="{}_{}".format(model_name, set)
                        )
                        .exists()
                    ).scalar():
                        for index, beam in enumerate(output):
                            sentence = clean_sentence(beam, vocab)
                            caption_index = str(index + 5)
                            captions_obj = ImageCaptions(
                                image_path=image_id,
                                caption_index=caption_index,
                                set="{}_{}".format(model_name, set),
                                caption=sentence,
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
                        for i in range(num_captions):
                            sentence = captions[i][k]
                            # print(caption)
                            # sentence = clean_sentence(caption.cpu().numpy(), vocab)
                            if model_name == "flickr":
                                caption_index = image_ids[k].split("#")[1]
                            else:
                                caption_index = i
                            # sentence = clean_sentence(captions[k].cpu().numpy(), vocab)
                            # caption_index = image_ids[k].split("#")[1]

                            if not db.session.query(
                                db.session.query(ImageCaptions)
                                .filter_by(
                                    image_path=image_id,
                                    caption_index=caption_index,
                                    set="{}_{}".format(model_name, set),
                                )
                                .exists()
                            ).scalar():
                                captions_obj = ImageCaptions(
                                    image_path=image_id,
                                    caption_index=caption_index,
                                    set="{}_{}".format(model_name, set),
                                    caption=sentence,
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
                                    "Alredy Exist: {}, {}".format(
                                        image_id, caption_index
                                    )
                                )
            return make_response(dict(status="Data Upload Success"), 200)
        except Exception as e:
            print(e)
            print(traceback.print_stack(e))
            return make_response(dict(status="Data Upload Failed"), 500)


@doc(
    summary="Compute Bleu Score", description="Compute Bleu Score", tags=["Captioning"]
)
class ComputeBleu(Resource):
    @marshal_with(BleuScoreSchema, code=200)
    def get(self, model_name, set, bleu_index):
        if model_name not in [
            "coco",
            "flickr",
            "flickr_attn",
            "coco_attn",
        ] or set not in ["test", "train", "val"]:
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
                unpadded_caption = d.caption.rstrip(" .")
                unpadded_caption += " ."
                if image_id not in original_captions:
                    original_captions[image_id] = []
                original_captions[image_id].append(
                    nltk.tokenize.word_tokenize(unpadded_caption.lower())
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
            candidate_corpus,
            reference_corpus,
            max_n=bleu_index,
            weights=[1.0 / bleu_index] * bleu_index,
        )
        return make_response(dict(bleu_Score=score), 200)


@doc(
    summary="Search Image given query",
    description="Search Image given query",
    tags=["Captioning"],
)
class SearchImage(Resource):
    @marshal_with(PopulateSearchSchema, code=200)
    def get(self, model_name, set, bleu_index, filter, query):
        config = Config()

        if model_name not in [
            "coco",
            "flickr",
            "flickr_attn",
            "coco_attn",
        ] or set not in ["test", "train", "val"] or filter not in ["True", "False"]:
            return make_response(dict(status="Invalid set or model name"), 500)

        data = (
            db.session.query(ImageCaptions)
            .filter_by(set="{}_{}".format(model_name, set))
            .all()
        )
        similarity_scores = {}
        for index, d in enumerate(data):
            caption_index = d.caption_index
            if caption_index >= 5:
                unpadded_caption = d.caption.rstrip(" .")
                unpadded_caption += " ."
                bleu_sc = [
                    [
                        bleu_score(
                            [self.filter_stopwords(nltk.tokenize.word_tokenize(query), filter)],
                            [[self.filter_stopwords(nltk.tokenize.word_tokenize(unpadded_caption), filter)]],
                            max_n=bleu_index,
                            weights=[1.0 / bleu_index] * bleu_index,
                        )
                    ]
                ]

                if d.image_path not in similarity_scores:
                    similarity_scores[d.image_path] = 0
                similarity_scores[d.image_path] = max(
                    bleu_sc[0][0], similarity_scores[d.image_path]
                )
        sorted_dict = {
            k: v
            for k, v in sorted(
                similarity_scores.items(), key=lambda item: item[1], reverse=True
            )
        }
        # print(sorted_dict)
        list_images = []
        count = 0
        for k, v in sorted_dict.items():
            print(f"BleuScore for Image: {k} is {v}")
            sys.stdout.flush()
            list_images.append(k)
            count += 1
            if count == 5:
                break
        return make_response(dict(image_ids=str(list_images)), 200)

    def filter_stopwords(self, tokens, filter="False"):
        if filter == "False":
            return tokens
        filtered = []
        for token in tokens:
            if token not in stopset:
                filtered.append(token)
        return filtered
