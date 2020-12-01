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

import math
import pickle
import traceback
import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords
from flask import make_response, request
from flask_apispec import marshal_with, doc, use_kwargs
from flask_apispec import MethodResource as Resource
from captioning.architecture.archpicker import get_encoder_decoder
from captioning.schema import PopulateImageSchema, BleuScoreSchema, SearchSchema
from captioning.captioning_config import Config
from captioning.data_handler.data_loader import get_data_loader, get_vocabulary
from captioning.main import get_device
from captioning.inference import beam_search
from models import ImageCaptions, db
import torch
from torchtext.data.metrics import bleu_score
from torch.autograd import Variable
from captioning.utils import convert_captions, clean_sentence
import sys
from bert.bert_encoder import BERT
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import requests, json

stopset = set(stopwords.words("english"))

config = Config()

@doc(
    summary="Load image captions",
    description="API end point to load image data from disk to the Database.",
    tags=["Captioning"],
)
class PopulateData(Resource):
    def initialize(self, model_set):
        config = Config()
        bert = BERT()
        data_loader = get_data_loader(config, mode=model_set, type=config.dataset_type)
        vocab = get_vocabulary(config, type=config.dataset_type, bert=bert)
        vocab_size = len(vocab)
        device = get_device()
        encoder, decoder = get_encoder_decoder(
            config.embed_size, config.hidden_size, vocab_size, bert=bert
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
        return data_loader, bert, vocab, config, device, encoder, decoder

    def check_db(self, image_path, model_set, caption_index=None):
        if caption_index is None:
            return db.session.query(
                db.session.query(ImageCaptions)
                .filter_by(image_path=image_path, set=model_set)
                .exists()
            ).scalar()
        else:
            return db.session.query(
                db.session.query(ImageCaptions)
                .filter_by(
                    image_path=image_path, caption_index=caption_index, set=model_set
                )
                .exists()
            ).scalar()

    def insert_db(self, image_id, caption_index, model_set, sentence):
        captions_obj = ImageCaptions(
            image_path=image_id,
            caption_index=caption_index,
            set=model_set,
            caption=sentence,
        )
        print("Inserting Prediction {}, {}".format(image_id, caption_index))
        db.session.add(captions_obj)
        db.session.commit()

    @marshal_with(PopulateImageSchema, code=200)
    def get(self, model_name, set):
        if model_name not in [
            "coco",
            "flickr",
            "flickr_attn",
            "flickr_attn_bert",
            "coco_attn",
        ] or set not in ["test", "train", "val"]:
            return make_response(dict(status="Invalid set or model name"), 500)

        data_loader, bert, vocab, config, device, encoder, decoder = self.initialize(
            set
        )
        model_set = f"{model_name}_{set}"

        total_step = math.ceil(
            len(data_loader.dataset) / data_loader.batch_sampler.batch_size
        )
        print("Total Steps: {}".format(total_step))

        try:
            for batch_index, (images, captions, image_ids) in enumerate(data_loader):
                print("step: {}".format(batch_index))
                images, encoded_captions, caption_lengths = convert_captions(
                    images, captions, vocab, config, bert=bert
                )
                if device:
                    images = images.cuda()

                if "flickr" in model_name:
                    captions = [captions]

                for img_index in range(images.shape[0]):
                    image = images[img_index].unsqueeze(0)
                    output = beam_search(encoder, decoder, image)

                    if "flickr" in model_name:
                        image_id = image_ids[img_index].split("#")[0]
                    else:
                        image_id = image_ids[img_index].item()

                    if not self.check_db(image_id, model_set):
                        for index, beam in enumerate(output):
                            sentence = clean_sentence(
                                beam, vocab, bert=bert, use_bert=config.enable_bert
                            )
                            caption_index = str(index + 5)
                            self.insert_db(image_id, caption_index, model_set, sentence)

                    else:
                        print("Already Exist: {}".format(image_id))

                    for cap_index in range(len(captions)):
                        sentence = captions[cap_index][img_index]
                        if "flickr" in model_name:
                            caption_index = image_ids[img_index].split("#")[1]
                        else:
                            caption_index = cap_index

                        if not self.check_db(image_id, model_set, caption_index):
                            self.insert_db(image_id, caption_index, model_set, sentence)
                        else:
                            print(
                                "Already Exist: {}, {}".format(image_id, caption_index)
                            )
            return make_response(dict(status="Data Upload Success"), 200)
        except Exception as e:
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
            "flickr_attn_bert",
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
    def get_embedding_layer(self, config, vocab_size, bert=None):
        encoder, decoder = get_encoder_decoder(
            config.embed_size, config.hidden_size, vocab_size, bert
        )
        if not torch.cuda.is_available():
            decoder.load_state_dict(
                torch.load(config.decoder_file, map_location=torch.device("cpu"))
            )
        else:
            decoder.load_state_dict(torch.load(config.decoder_file))

        return decoder.embedding

    def get_token_ids(self, query, vocab, is_filter=False, bert=None):
        caption = []
        importance = []
        tokens = self.filter_stopwords(
            nltk.tokenize.word_tokenize(str(query).lower()), is_filter
        )

        frequency = None
        if config.enable_bert:
            with open(config.vocab_file, "rb") as f:
                vocab = pickle.load(f)
                frequency = vocab.frequency

            tokenizer = bert.get_tokenizer()
            caption.extend(
                [tokenizer.convert_tokens_to_ids(token) for token in tokens]
            )
        else:
            frequency = vocab.frequency
            caption.extend([vocab(token) for token in tokens])

        for token in tokens:
            if token not in frequency.keys():
                importance.append(0.0)
            else:
                importance.append(1.0/frequency[token])

        return torch.tensor(caption), importance

    @marshal_with(SearchSchema, code=200)
    def get(self, model_name, set, bleu_index, filter, query):
        if (
            model_name
            not in ["coco", "flickr", "flickr_attn", "flickr_attn_bert", "coco_attn"]
            or set not in ["test", "train", "val"]
            or filter not in ["True", "False"]
        ):
            return make_response(dict(status="Invalid set or model name"), 500)

        config = Config()
        bert = BERT()
        vocab = get_vocabulary(config, type=config.dataset_type, bert=bert)
        embedding_layer = self.get_embedding_layer(config, len(vocab), bert=bert)

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
                token_id_query, importance = self.get_token_ids(query, vocab, filter, bert=bert)
                token_id_caption, _ = self.get_token_ids(unpadded_caption, vocab, filter, bert=bert)
                query_embedding = embedding_layer(token_id_query) / torch.norm(
                    embedding_layer(token_id_query), p=2, dim=1
                ).unsqueeze(1)
                caption_embedding = embedding_layer(token_id_caption) / torch.norm(
                    embedding_layer(token_id_caption), p=2, dim=1
                ).unsqueeze(1)
                similarity_val = torch.sum(
                    torch.max(
                        torch.matmul(query_embedding, caption_embedding.t()), dim=1
                    ).values * torch.tensor(importance)
                )

                # similarity_val = torch.mean(
                #     torch.matmul(query_embedding, caption_embedding.t())* torch.tensor(importance).unsqueeze(1)
                # )

                # similarity_val = torch.mean(
                #         F.relu(torch.matmul(caption_embedding, query_embedding.t()))
                # )
                if d.image_path not in similarity_scores:
                    similarity_scores[d.image_path] = 0
                similarity_scores[d.image_path] = similarity_val + similarity_scores[d.image_path]

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
            print(f"SimilarityScore for Image: {k} is {v}")
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

@doc(
    summary="Search Image given image",
    description="Search Image given image",
    tags=["Captioning"],
)
class SearchByImage(Resource):
    def initialize(self, model_set):
        config = Config()
        bert = BERT()
        data_loader = get_data_loader(config, mode=model_set, type=config.dataset_type)
        vocab = get_vocabulary(config, type=config.dataset_type, bert=bert)
        vocab_size = len(vocab)
        device = get_device()
        encoder, decoder = get_encoder_decoder(
            config.embed_size, config.hidden_size, vocab_size, bert=bert
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
        return data_loader, bert, vocab, config, device, encoder, decoder

    def get_embedding_layer(self, config, vocab_size, bert=None):
        encoder, decoder = get_encoder_decoder(
            config.embed_size, config.hidden_size, vocab_size, bert
        )
        if not torch.cuda.is_available():
            decoder.load_state_dict(
                torch.load(config.decoder_file, map_location=torch.device("cpu"))
            )
        else:
            decoder.load_state_dict(torch.load(config.decoder_file))

        return decoder.embedding

    def get_token_ids(self, query, vocab, is_filter=False, bert=None):
        caption = []
        importance = []
        tokens = self.filter_stopwords(
            nltk.tokenize.word_tokenize(str(query).lower()), is_filter
        )

        frequency = None
        if config.enable_bert:
            with open(config.vocab_file, "rb") as f:
                vocab = pickle.load(f)
                frequency = vocab.frequency

            tokenizer = bert.get_tokenizer()
            caption.extend(
                [tokenizer.convert_tokens_to_ids(token) for token in tokens]
            )
        else:
            frequency = vocab.frequency
            caption.extend([vocab(token) for token in tokens])

        for token in tokens:
            if token not in frequency.keys():
                importance.append(0.0)
            else:
                importance.append(1.0/frequency[token])

        return torch.tensor(caption), importance

    @marshal_with(SearchSchema, code=200)
    def get(self, model_name, set, bleu_index, filter, image_path):
        if (
            model_name
            not in ["coco", "flickr", "flickr_attn", "flickr_attn_bert", "coco_attn"]
            or set not in ["test", "train", "val"]
            or filter not in ["True", "False"]
        ):
            return make_response(dict(status="Invalid set or model name"), 500)

        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        def image_loader(image_name):
            """load image, returns cuda tensor"""
            image = Image.open(image_name)
            image = transform(image).float()
            image = Variable(image, requires_grad=True)
            image = image.unsqueeze(0)
            return image

        config = Config()
        image_path = os.path.join(config.image_search_dir, image_path)
        print(image_path)

        data_loader, bert, vocab, config, device, encoder, decoder = self.initialize(
            set
        )

        model_set = f"{model_name}_{set}"
        image = image_loader(image_path)
        vocab = get_vocabulary(config, type=config.dataset_type, bert=bert)
        # embedding_layer = self.get_embedding_layer(config, len(vocab), bert=bert)
        embedding_layer = decoder.embedding
        device = get_device()

        if device:
            image = image.cuda()

        output = beam_search(encoder, decoder, image)
        for index, s in enumerate(output):
            sentence = clean_sentence(s, vocab, bert=bert, use_bert=config.enable_bert)
            print("Predicted Caption {}: ".format(index) + str(sentence))

        query = clean_sentence(output[0], vocab, bert=bert, use_bert=config.enable_bert)
        print(f"query: {query}")

        url = f"http://0.0.0.0:5000/search/{model_name}/{set}/{bleu_index}/{filter}/{query.replace(' ', '%20')}"
        response = requests.get(url)
        json_resp = json.loads(response.text)
        list_images = [im.strip("\"\'\[\]") for im in json_resp["image_ids"].split(', ')]

        return make_response(dict(image_ids=str(list_images)), 200)

    def filter_stopwords(self, tokens, filter="False"):
        if filter == "False":
            return tokens
        filtered = []
        for token in tokens:
            if token not in stopset:
                filtered.append(token)
        return filtered
