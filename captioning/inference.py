from captioning.config import Config
from captioning.data_handler.data_loader import get_data_loader
from captioning.utils import imshow


def clean_sentence(output, test_loader):
    words_sequence = []

    for i in output:
        if i == 1:
            continue
        words_sequence.append(test_loader.dataset.vocab.idx2word[i])

    words_sequence = words_sequence[1:-1]
    sentence = " ".join(words_sequence)
    sentence = sentence.capitalize()

    return sentence


def get_predict(image, encoder, decoder, test_loader):
    # image = image.to(device)
    imshow(image)
    features = encoder(image).unsqueeze(1)
    output = decoder.sample(features)
    sentence = clean_sentence(output, test_loader)
    print(sentence)
