import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import math
from data_loader import get_loader
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN, EncoderCNNATT, DecoderRNNATT
from adaptive_model import Encoder2Decoder
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Image preprocessing
    # For normalization, see https://github.com/pytorch/vision#models
    transform = transforms.Compose([
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper.
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab,
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)

    # Build the models
    adaptive_model = Encoder2Decoder(args.embed_size, len(vocab), args.hidden_size)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    # params = list(decoder.parameters())
    params = list(adaptive_model.encoder.affine_a.parameters()) + list(adaptive_model.encoder.affine_b.parameters()) \
             + list(adaptive_model.decoder.parameters())

    if torch.cuda.is_available():
        adaptive_model.cuda()
        criterion.cuda()
        # encoder.cuda()
        # decoder.cuda()

    cnn_subs = list(adaptive_model.encoder.resnet.children())[args.fine_tune_start_layer:]
    cnn_params = [list(sub_module.parameters()) for sub_module in cnn_subs]
    cnn_params = [item for sublist in cnn_params for item in sublist]
    cnn_optimizer = torch.optim.Adam(cnn_params, lr=args.lr_cnn,
                                     betas=(args.alpha, args.beta))

    print(len(data_loader), 'data')

    # Train the Models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        learning_rate = args.lr
        if epoch > args.lr_decay:
            frac = float(epoch - args.lr_decay) / args.learning_rate_decay_every
            decay_factor = math.pow(0.5, frac)

            # Decay the learning rate
            learning_rate = args.lr * decay_factor

        print('Learning Rate for Epoch %d: %.6f' % (epoch, learning_rate))

        optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(args.alpha, args.beta))

        for i, (images, captions, lengths) in enumerate(data_loader):
            # Set mini-batch dataset
            # print images.size()
            images = to_var(images,volatile=True)
            captions = to_var(captions)
            lengths = [ cap_len - 1  for cap_len in lengths ]
            targets = pack_padded_sequence(captions[:, 1:], lengths, batch_first=True)[0]
            #print(lengths,'len')
            adaptive_model.train()
            adaptive_model.zero_grad()

            packed_scores = adaptive_model(images, captions, lengths)
            print(packed_scores[0].size(),targets.size())
            print(packed_scores[0].requires_grad)
            loss = criterion(packed_scores[0], targets)
            loss.backward()
            for p in adaptive_model.decoder.lstm.parameters():
                p.data.clamp_(-args.clip, args.clip)

            optimizer.step()

            if epoch > args.cnn_epoch:
                cnn_optimizer.step()


            # Print log info
            if i % args.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                      % (epoch, args.num_epochs, i, total_step,
                         loss.data[0], np.exp(loss.data[0])))

                # Save the models
            if (i + 1) % args.save_step == 0:
                torch.save(adaptive_model.state_dict(),
                           os.path.join(args.model_path,
                                        'adaptive_model-%d-%d.pkl' % (epoch + 1, i + 1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/',
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='./data/val_resized2014',
                        help='directory for resized images')
    parser.add_argument('--caption_path', type=str,
                        default='./data/annotations/captions_val2014.json',
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int, default=10,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000,
                        help='step size for saving trained models')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=256,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)

    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='alpha in Adam')
    parser.add_argument('--beta', type=float, default=0.999,
                        help='beta in Adam')
    parser.add_argument('--lr_cnn', type=float, default=1e-4,
                        help='learning rate for fine-tuning CNN')

    parser.add_argument('--cnn_epoch', type=int, default=4,
                        help='start fine-tuning CNN after')
    parser.add_argument('--fine_tune_start_layer', type=int, default=5,
                        help='CNN fine-tuning layers from: [0-7]')

    parser.add_argument('--clip', type=float, default=0.1)
    parser.add_argument('--lr_decay', type=int, default=20, help='epoch at which to start lr decay')
    parser.add_argument('--learning_rate_decay_every', type=int, default=50,
                        help='decay learning rate at every this number')

    args = parser.parse_args()
    print(args)
    main(args)

