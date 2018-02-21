import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import bleu
import os
import nltk.translate.bleu_score as BLEU
from torch.autograd import Variable
from torchvision import transforms
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN, EncoderCNNATT,DecoderRNNATT
from PIL import Image
from data_loader import get_loader
from torch.nn.utils.rnn import pack_padded_sequence

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def translate(captions,vocab):
    caps_id = captions.data.numpy()
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
    encoder = EncoderCNN(args.embed_size)
    encoder.eval()  # evaluation mode (BN uses moving mean/variance)

    #att
    encoder_att = EncoderCNNATT(args.embed_size)
    encoder_att.eval()
    decoder_att = DecoderRNNATT(args.embed_size, args.hidden_size,
                         len(vocab), args.num_layers)
    decoder_att.eval()

    decoder = DecoderRNN(args.embed_size, args.hidden_size,
                         len(vocab), args.num_layers)

    # Load the trained model parameters
    #print args.encoder_path
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    #att
    #encoder_att.load_state_dict(torch.load(args.encoder_path))

    #data_loader = get_loader(args.image_dir, args.caption_path, vocab,
    #                         transform, args.batch_size,
    #                         shuffle=True, num_workers=args.num_workers)

    data_loader = get_loader(args.image_dir, args.caption_path, vocab,
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)



    print len(data_loader),'data'

    total_num = len(data_loader)*args.batch_size
    print total_num
    num_correct=0
    tested=0

    hypotheses=[]
    references=[]

    for i, (images, captions, lengths) in enumerate(data_loader):
        print images.size(),'s'
        tested += args.batch_size
        if i==1:
            break;

        # If use gpu
        if torch.cuda.is_available():
            encoder.cuda()
            decoder.cuda()

        # Prepare Image
        images = to_var(images, volatile=True)
        captions = to_var(captions)
        #targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
        max_sent_length=captions[-1].size(0)
        print max_sent_length,'length'
        print captions.size(),'caption_size'
        #print captions[0].size()
        #print captions[0]
        #print targets.size()
        #print captions[:,0]
        print images.size(),'im'
        # Generate caption from image
        #features=encoder(images)
        #features = encoder(images)
        #print features.size(),'fe'
        features_att=encoder_att(images)
        print features_att.size(),'fet'
        #sampled_cap_att=decoder.sample_att(features_att,features,max_sent_length)

        #sampled_captions = decoder.sample(features,max_sent_length)
        #targets=torch.transpose(sampled_captions.view(max_sent_length,-1),0,1);
        targets=[]

        print targets.size(),'ans'
        #print targets
        #print captions
        ref_sents=translate(captions,vocab)
        hypo_sents=translate(targets,vocab)

        references.extend(ref_sents)
        hypotheses.extend(hypo_sents)
        num_correct_t = targets.data.eq(captions.data).sum()
        print num_correct_t,'num correct'
        num_correct += num_correct_t


        #feature = encoder(image_tensor)
        #sampled_ids = decoder.sample(feature)
        #sampled_ids = sampled_ids.cpu().data.numpy()

        # Decode word_ids to words
        #sampled_caption = []
        #for word_id in sampled_ids:
        #    word = vocab.idx2word[word_id]
        #    sampled_caption.append(word)
        #    if word == '<end>':
        #        break
        #sentence = ' '.join(sampled_caption)

        # Print out image and generated caption.
        #print (sentence)


    hypo_ref_out=(hypotheses,references)
    with open('hypo_out.txt', 'wb') as handle:
        pickle.dump(hypo_ref_out,handle)
    print len(hypotheses)
    print hypotheses[0:10]
    print references[0:10]
    bleu_score=bleu.BLEU(hypotheses,[references])
    print bleu_score

    print 'num_correct',num_correct,'total',tested,total_num
    score = BLEU.corpus_bleu(references,hypotheses)
    score1 = BLEU.corpus_bleu(references,hypotheses,weights=[1,0,0,0])
    score2 = BLEU.corpus_bleu(references, hypotheses,weights=[0.5,0.5,0,0])
    score3 = BLEU.corpus_bleu(references, hypotheses,weights=[0.3,0.3,0.3,0])
    score4 = BLEU.corpus_bleu(references, hypotheses, weights=[0.25,0.25,0.25,0.25])
    print score,score1,score2,score3,score4



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_path', type=str, default='./models/encoder-5-3000.pkl',
                        help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='./models/decoder-5-3000.pkl',
                        help='path for trained decoder')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='./data/val_resized2014',
                        help='directory for dev resized images')
    parser.add_argument('--caption_path', type=str,
                        default='./data/annotations/captions_val2014.json',
                        help='path for dev annotation json file')
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int, default=256,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in lstm')
    args = parser.parse_args()
    main(args)