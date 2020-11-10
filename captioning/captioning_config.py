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


class CaptioningConfig:
    """Base config."""

    dataset_type = "coco"  # Supported types: 'flickr8k', 'coco'
    data_dir = os.path.join(ROOT_DIR, "captioning/data")
    models_dir = os.path.join(ROOT_DIR, "captioning/models")
    dummy_annotations_file = os.path.join(
        data_dir, "Flickr8k_text/Flickr8k_dummy.token.txt"
    )
    encoder_prefix = 'encoder_30k_naive'
    decoder_prefix = 'decoder_30k_naive'
    annotations_file = os.path.join(data_dir, "Flickr8k_text/Flickr8k.token.txt")
    images_dir = os.path.join(data_dir, "Flickr8k_Dataset")
    train_id_file = os.path.join(data_dir, "Flickr8k_text/Flickr_8k.trainImages.txt")
    val_id_file = os.path.join(data_dir, "Flickr8k_text/Flickr_8k.devImages.txt")
    test_id_file = os.path.join(data_dir, "Flickr8k_text/Flickr_8k.testImages.txt")
    test_root_dir_coco = "/Users/shubhi/GT_courses/DeepLearning/project/coco_captioning/test2017"
    #test_root_dir_coco = "/home/shubhi_agl22/datasets/coco_captioning/test2017"
    train_root_dir_coco = "/home/shubhi_agl22/datasets/coco_captioning/train2017"
    val_root_dir_coco = "/home/shubhi_agl22/datasets/coco_captioning/val2017"
    train_ann_file_coco = "/home/shubhi_agl22/datasets/coco_captioning/annotations/captions_train2017.json"
    val_ann_file_coco = "/home/shubhi_agl22/datasets/coco_captioning/annotations/captions_val2017.json"
    #test_ann_file_coco = "/home/shubhi_agl22/datasets/coco_captioning/annotations/image_info_test2017.json"
    test_ann_file_coco = "/Users/shubhi/GT_courses/DeepLearning/project/coco_captioning/image_info_test2017.json"
    vocab_file = os.path.join(data_dir, "vocab_{}.pkl".format(dataset_type))
    encoder_file = os.path.join(models_dir, "encoder_30k_naive-2.pth")
    decoder_file = os.path.join(models_dir, "decoder_30k_naive-2.pth")
    batch_size = 8  # batch size
    vocab_threshold = 5  # minimum word count threshold
    vocab_from_file = True  # if True, load existing vocab file
    embed_size = 300  # dimensionality of image and word embeddings
    hidden_size = 512  # number of features in hidden state of the RNN decoder
    epoch_range = range(1, 20)  # number of training epochs
    save_every = 1  # determines frequency of saving model weights
    print_every = 10  # determines window for printing average loss
    num_workers = 2
    load_from_file = False
    run_training = False
    run_prediction = True
    learning_rate = 0.001
    scheduler_gamma = 0.95
    momentum = 0.9
    weight_decay = 0.99  # l2 norm strength
    log_file = "training.log"  # name of file with saved training loss and perplexity
    beam_size = 3
    max_length = 40
    verbose = True
