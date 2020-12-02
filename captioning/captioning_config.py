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

import os
from config import ROOT_DIR


class BaseConfig:
    def __init__(self):
        self.dataset_type = "flickr8k"                                # Set dataset type [flickr, coco]
        self.models_dir = os.path.join(ROOT_DIR, "captioning/models") # Set directory to save models
        self.data_dir = os.path.join(ROOT_DIR, "captioning/data/")    # Set directory to store train/test/val data
        self.enable_wandb = False                                     # Switch to enable/diable wandb logging
        self.verbose = False                                          # Verbose Flag to print stats
        self.enable_bert = True                                       # enable bert embeddings in attention model
        self.arch_name = "attention"                                  # Set architecture type [vanilla, attention]


class COCO_Config(BaseConfig):
    def __init__(self):
        super(COCO_Config, self).__init__()
        self.test_root_dir_coco = os.path.join(self.data_dir, "coco/test2017")   # COCO test data directory
        self.train_root_dir_coco = os.path.join(self.data_dir, "coco/train2017") # COCO train data directory
        self.val_root_dir_coco = os.path.join(self.data_dir, "coco/val2017")     # COCO Validation data directory
        self.train_ann_file_coco = os.path.join(
            self.data_dir, "coco/captions_train2017.json"                        # COCO training annotations file.
        )
        self.val_ann_file_coco = os.path.join(
            self.data_dir, "coco/captions_val2017.json"                          # COCO validation annotations file.
        )
        self.test_ann_file_coco = os.path.join(
            self.data_dir, "coco/image_info_test2017.json"                       # COCO test annotations file.
        )
        self.coco_subset_size = 30000                                            # Set subset size to sample from COCO
                                                                                 # for training


class Flickr_Config(BaseConfig):
    def __init__(self):
        super(Flickr_Config, self).__init__()
        self.annotations_file = os.path.join(
            self.data_dir,
            "flickr/Flickr8k_text/Flickr8k.token.txt"               # Flickr annotations file.
        )
        self.images_dir = os.path.join(self.data_dir,
                                       "flickr/Flicker8k_Dataset")  # Set Flickr data directory
        self.train_id_file = os.path.join(
            self.data_dir,
            "flickr/Flickr8k_text/Flickr_8k.trainImages.txt"        # Flickr id file with train images.
        )
        self.val_id_file = os.path.join(
            self.data_dir,
            "flickr/Flickr8k_text/Flickr_8k.devImages.txt"          # Flickr id file with val images.
        )
        self.test_id_file = os.path.join(
            self.data_dir,
            "flickr/Flickr8k_text/Flickr_8k.testImages.txt"         # Flickr id file with test images.
        )
        self.flickr_subsample = False                               # Switch to enable sampling a subset.
        self.flickr_subset_size_train = 50                          # Subset size for train data.
        self.flickr_subset_size_val = 10                            # subset size for validation data.


class Config(Flickr_Config, COCO_Config):
    """Base config."""

    def __init__(self):
        super(Config, self).__init__()
        self.image_search_dir = os.path.join(
            self.data_dir, "imagesearch"
        )                                                           # Image Search Directory path
        self.encoder_prefix = "encoder_flickr_attn_bert"            # Prefix for encoder model
        self.decoder_prefix = "decoder_flickr_attn_bert"            # Prefix for decoder model
        self.encoder_file = os.path.join(
            self.models_dir, "{}-19.pth".format(self.encoder_prefix)
        )                                                           # Set encoder model path
        self.decoder_file = os.path.join(
            self.models_dir, "{}-19.pth".format(self.decoder_prefix)
        )                                                           # Set decoder model path
        self.vocab_file = os.path.join(
            self.data_dir, "vocab_{}.pkl".format(self.dataset_type)
        )                                                           # Set vocab file path
        self.batch_size = 128                                       # batch size
        self.vocab_threshold = 5                                    # minimum word count threshold
        self.vocab_from_file = True                                 # switch to toggle loading existing vocab file
        self.embed_size = 512                                       # dimensionality of image and word embeddings
        if self.enable_bert:
            self.embed_size = 768
        self.hidden_size = 512                                      # number of features in hidden state of decoder
        self.epoch_range = range(1, 20)                             # range of training epochs
        self.save_every = 1                                         # frequency of saving model weights
        self.print_every = 10                                       # frequency for printing average loss
        self.num_workers = 0                                        # Number of worker in data loader
        self.load_from_file = False                                 # switch to toggle loading existing model.
        self.run_training = True                                    # Run Train
        self.run_prediction = False                                 # Run Predict
        self.learning_rate = 0.001                                  # Learning Rate
        self.scheduler_gamma = 0.95                                 # learning rate decay for scheduler
        self.momentum = 0.9                                         # Momentum for ADAM optimizer
        self.weight_decay = 0.99                                    # l2 norm strength
        self.log_file = (
            "training.log"
        )                                                           # Log File Name
        self.beam_size = 5                                          # Beam Size for Beam Search
        self.max_length = 40                                        # Limit of words in a caption.
        self.train_encoder = False                                  # Enable Backpropagation on Encoder
        self.features_img_size = 2048                               # Size of out feature vector from Encoder
        self.attn_size = 512                                        # Size of attention vector
        self.encoded_img_size = 14                                  # Size of image to pass through attention layer.
        self.dropout = 0.5                                          # Dropout
        self.max_char_length = 150                                  # Limit of characters in a caption.
        self.use_bleu = False                                       # Use BLEU score for similarity search. Set False to use cosine similarity.