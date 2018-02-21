import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
#import bleu
import os
import nltk.translate.bleu_score as BLEU
from torch.autograd import Variable
from torchvision import transforms
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN, EncoderCNNATT,DecoderRNNATT
from adaptive_model import Encoder2Decoder
from PIL import Image
from data_loader import get_loader
from torch.nn.utils.rnn import pack_padded_sequence

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image


def translate(captions,vocab):
    caps_id = captions.cpu().data.numpy()
    sents=[]
    for cap_id in caps_id:
        sentence = []
        for word_id in cap_id:
            #print word_id
            word = vocab.idx2word[word_id]
            sentence.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sentence)
        sents.append(sentence)
    return sents

def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    #print (vocab.word2idx['<start>'])
    # Build Models
    model = Encoder2Decoder(args.embed_size, len(vocab), args.hidden_size)


    # Load the trained model parameters
    model.load_state_dict(torch.load(args.model_path))

    max_sent_length = 20
    # If use gpu
    if torch.cuda.is_available():
        model.cuda()

    # Prepare Image
    image = load_image(args.image, transform)
    image = to_var(image, volatile=True)
    #image = image.unsqueeze(0)
    #targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

    answers,attention,beta = model.sample(image,max_sent_length)
    print translate(answers,vocab)
    print attention
    print beta






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/adaptive_model-8-1000.pkl',
                        help='path for trained model')
    parser.add_argument('--image', type=str, required=True,
                        help='input image for generating caption')
    parser.add_argument('--image_dir', type=str, default='./data/val_resized2014',
                        help='directory for dev resized images')
    parser.add_argument('--caption_path', type=str,
                        default='./data/annotations/captions_val2014.json',
                        help='path for dev annotation json file')
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                        help='path for vocabulary wrapper')
        

    parser.add_argument('--embed_size', type=int, default=256,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in lstm')
    args = parser.parse_args()
    main(args)
