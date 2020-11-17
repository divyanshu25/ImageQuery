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

import os
from config import ROOT_DIR


class BaseConfig:
    def __init__(self):
        self.dataset_type = "flickr8k"  # Supported types: 'flickr8k', 'coco'
        self.models_dir = os.path.join(ROOT_DIR, "captioning/models")
        self.data_dir = os.path.join(ROOT_DIR, "captioning/data/")
        self.enable_wandb = False
        self.verbose = True


class COCO_Config(BaseConfig):
    def __init__(self):
        super(COCO_Config, self).__init__()
        self.test_root_dir_coco = os.path.join(self.data_dir, "coco/test2017")
        self.train_root_dir_coco = os.path.join(self.data_dir, "coco/train2017")
        self.val_root_dir_coco = os.path.join(self.data_dir, "coco/val2017")
        self.train_ann_file_coco = os.path.join(
            self.data_dir, "coco/captions_train2017.json"
        )
        self.val_ann_file_coco = os.path.join(
            self.data_dir, "coco/captions_val2017.json"
        )
        self.test_ann_file_coco = os.path.join(
            self.data_dir, "coco/image_info_test2017.json"
        )
        self.coco_subset_size = 30000


class Flickr_Config(BaseConfig):
    def __init__(self):
        super(Flickr_Config, self).__init__()
        self.annotations_file = os.path.join(
            self.data_dir, "flickr/Flickr8k_text/Flickr8k.token.txt"
        )
        self.images_dir = os.path.join(self.data_dir, "flickr/Flicker8k_Dataset")
        self.train_id_file = os.path.join(
            self.data_dir, "flickr/Flickr8k_text/Flickr_8k.trainImages.txt"
        )
        self.val_id_file = os.path.join(
            self.data_dir, "flickr/Flickr8k_text/Flickr_8k.devImages.txt"
        )
        self.test_id_file = os.path.join(
            self.data_dir, "flickr/Flickr8k_text/Flickr_8k.testImages.txt"
        )
        self.flickr_subsample = False
        self.flickr_subset_size_train = 50
        self.flickr_subset_size_val = 10


class Config(Flickr_Config, COCO_Config):
    """Base config."""

    def __init__(self):
        super(Config, self).__init__()
        self.encoder_prefix = "encoder_flickr_attn"
        self.decoder_prefix = "decoder_flickr_attn"
        self.encoder_file = os.path.join(
            self.models_dir, "{}-19.pth".format(self.encoder_prefix)
        )
        self.decoder_file = os.path.join(
            self.models_dir, "{}-19.pth".format(self.decoder_prefix)
        )
        self.vocab_file = os.path.join(
            self.data_dir, "vocab_{}.pkl".format(self.dataset_type)
        )
        self.batch_size = 128  # batch size
        self.vocab_threshold = 5  # minimum word count threshold
        self.vocab_from_file = True  # if True, load existing vocab file
        self.embed_size = 512  # dimensionality of image and word embeddings
        self.hidden_size = 512  # number of features in hidden state of the RNN decoder
        self.epoch_range = range(1, 20)  # number of training epochs
        self.save_every = 1  # determines frequency of saving model weights
        self.print_every = 10  # determines window for printing average loss
        self.num_workers = 0
        self.load_from_file = False
        self.run_training = False
        self.run_prediction = True
        self.learning_rate = 0.001
        self.scheduler_gamma = 0.95
        self.momentum = 0.9
        self.weight_decay = 0.99  # l2 norm strength
        self.log_file = (
            "training.log"
        )  # name of file with saved training loss and perplexity
        self.beam_size = 5
        self.max_length = 40
        self.train_encoder = False
        self.arch_name = "attention"
        self.features_img_size = 2048
        self.attn_size = 512
        self.encoded_img_size = 14
        self.dropout = 0.5
        self.max_char_length = 150
