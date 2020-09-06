import os


class Config:
    """Base config."""

    data_dir = "/Users/dibu/Gatech Academics/ImageQuery/captioning/data/Flickr_Data/Flickr_Data"
    annotations_file = os.path.join(data_dir, "Flickr_TextData/Flickr8k.token.txt")
    images_dir = os.path.join(data_dir, "Images")
    train_id_file = os.path.join(data_dir, "Flickr_TextData/Flickr_8k.trainImages.txt")
    val_id_file = os.path.join(data_dir, "Flickr_TextData/Flickr_8k.devImages.txt")
    test_id_file = os.path.join(data_dir, "Flickr_TextData/Flickr_8k.testImages.txt")
    vocab_file = "/Users/dibu/Gatech Academics/ImageQuery/captioning/data/vocab.pkl"
    batch_size = 64  # batch size
    vocab_threshold = 2  # minimum word count threshold
    vocab_from_file = True  # if True, load existing vocab file
    embed_size = 300  # dimensionality of image and word embeddings
    hidden_size = 512  # number of features in hidden state of the RNN decoder
    num_epochs = 3  # number of training epochs
    save_every = 1  # determines frequency of saving model weights
    print_every = 100  # determines window for printing average loss
    num_workers = 1
    log_file = (
        "training_log.txt"  # name of file with saved training loss and perplexity
    )
