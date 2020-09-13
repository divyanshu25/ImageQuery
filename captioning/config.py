import os


class Config:
    """Base config."""

    data_dir = "./data"
    annotations_file = os.path.join(data_dir, "Flickr8k_text/Flickr8k.token.txt")
    images_dir = os.path.join(data_dir, "Flickr8k_Dataset")
    train_id_file = os.path.join(data_dir, "Flickr8k_text/Flickr_8k.trainImages.txt")
    val_id_file = os.path.join(data_dir, "Flickr8k_text/Flickr_8k.devImages.txt")
    test_id_file = os.path.join(data_dir, "Flickr8k_text/Flickr_8k.testImages.txt")
    vocab_file = "./data/vocab.pkl"
    encoder_file = "./models/encoder-1.pth"
    decoder_file = "./models/decoder-1.pth"
    batch_size = 1  # batch size
    vocab_threshold = 2  # minimum word count threshold
    vocab_from_file = True  # if True, load existing vocab file
    embed_size = 300  # dimensionality of image and word embeddings
    hidden_size = 512  # number of features in hidden state of the RNN decoder
    num_epochs = 1  # number of training epochs
    save_every = 1  # determines frequency of saving model weights
    print_every = 1  # determines window for printing average loss
    num_workers = 1
    load_from_file = False
    do_train = False
    get_prediction = True
    learning_rate = 0.001
    momentum = 0.9
    log_file = (
        "training_log.txt"  # name of file with saved training loss and perplexity
    )
