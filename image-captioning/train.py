import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN,EncoderCNNATT,DecoderRNNATT
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
    #encoder = EncoderCNN(args.embed_size)
    #decoder = DecoderRNN(args.embed_size, args.hidden_size,
    #                     len(vocab), args.num_layers)
    #encoder=EncoderCNNATT(args.embed_size)
    #decoder=DecoderRNNATT(args.embed_size, args.hidden_size,
    #                   len(vocab), args.num_layers)

    adaptive_model = Encoder2Decoder(args.embed_size,len(vocab),args.hidden_size)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    #params = list(decoder.parameters())
    params = list(adaptive_model.encoder.affine_a.parameters())+list(adaptive_model.encoder.affine_b.parameters()) \
                + list(adaptive_model.decoder.parameters())

    if torch.cuda.is_available():
        adaptive_model.cuda()
        criterion.cuda()
        # encoder.cuda()
        # decoder.cuda()

    cnn_subs = list(adaptive.encoder.resnet_conv.children())[args.fine_tune_start_layer:]
    cnn_params = [list(sub_module.parameters()) for sub_module in cnn_subs]
    cnn_params = [item for sublist in cnn_params for item in sublist]
    cnn_optimizer = torch.optim.Adam(cnn_params, lr=args.learning_rate_cnn,
                                     betas=(args.alpha, args.beta))


    print len(data_loader),'data'

    # Train the Models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):

        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(args.alpha, args.beta))

        for i, (images, captions, lengths) in enumerate(data_loader):
            # Set mini-batch dataset
            #print images.size()
            images = to_var(images, volatile=True)
            captions = to_var(captions)
            targets = pack_padded_sequence(captions[:,1:], lengths, batch_first=True)[0]
            #print captions.size()
            # Forward, Backward and Optimize
            #decoder.zero_grad()
            #encoder.zero_grad()
            #print(images.size(),'imsize')
             #features = encoder(images)
            #print(features.size(),'fsize')
            #outputs = decoder(features, captions,lengths)
            #print(outputs.volatile)
            #loss = criterion(outputs, targets)

            adaptive_model.train()
            adaptive_model.zero_grad()
            packed_scores = adaptive_model(images,captions,lengths)
            loss = criterion(packed_scores[0],captions)
            loss.backward()
            for params in adaptive_model.decoder.lstm.parameters():
                params.data.clamp_(-args.clip,args.clip)


            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                      %(epoch, args.num_epochs, i, total_step, 
                        loss.data[0], np.exp(loss.data[0]))) 
                
            # Save the models
            if (i+1) % args.save_step == 0:
                #torch.save(decoder.state_dict(),
                #           os.path.join(args.model_path,
                #                        'decoder-%d-%d.pkl' %(epoch+1, i+1)))
                #torch.save(encoder.state_dict(),
                #           os.path.join(args.model_path,
                #                        'encoder-%d-%d.pkl' %(epoch+1, i+1)))
                torch.save(adaptive_model.state_dict(),
                           os.path.join(args.model_path,
                                        'adaptive_model-%d-%d.pkl' %(epoch+1,i+1)))


                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/' ,
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 ,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='./data/val_resized2014' ,
                        help='directory for resized images')
    parser.add_argument('--caption_path', type=str,
                        default='./data/annotations/captions_val2014.json',
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=10,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000,
                        help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256 ,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512 ,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)