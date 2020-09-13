from captioning.config import Config
from captioning.data_handler.data_loader import get_data_loader
from captioning.utils import imshow, clean_sentence


def get_predict(image, encoder, decoder, test_loader):
    #image = image.to(device)
    print(image.shape)
    features = encoder(image).unsqueeze(1)
    output = decoder.sample(features)
    sentence = clean_sentence(output, test_loader)
    print(sentence)
    imshow(image[0])

