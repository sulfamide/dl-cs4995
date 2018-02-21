import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
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
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),                
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
    att_beta=[]
    hypotheses=[]
    references=[]
    max_sent_length =20
    for i, (images, captions, lengths) in enumerate(data_loader):
        tested += args.batch_size
        
        # If use gpu
        if torch.cuda.is_available():
            model.cuda()

        # Prepare Image
        images = to_var(images, volatile=True)
        captions = to_var(captions,volatile=True)
        #targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        answers,attention,beta = model.sample(images,max_sent_length)
        #print answers
        att_beta=[answers,attention,beta]
        with open('att_beta.txt','ab') as hand:
                    pickle.dump(att_beta,hand)
        targets=answers
        
        print targets.size(),'ans'
        #print targets
        #print captions
        ref_sents=translate(captions,vocab)
        hypo_sents=translate(targets,vocab)

        references.extend(ref_sents)
        hypotheses.extend(hypo_sents)

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
 
    with open('hypo_out_att.txt', 'wb') as handle:
        pickle.dump(hypo_ref_out,handle)
    print len(hypotheses)
    
    score = BLEU.corpus_bleu(references,hypotheses)
    score1 = BLEU.corpus_bleu(references,hypotheses,weights=[1,0,0,0])
    score2 = BLEU.corpus_bleu(references, hypotheses,weights=[0.5,0.5,0,0])
    score3 = BLEU.corpus_bleu(references, hypotheses,weights=[0.3,0.3,0.3,0])
    score4 = BLEU.corpus_bleu(references, hypotheses, weights=[0.25,0.25,0.25,0.25])
    print score,score1,score2,score3,score4



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop_size', type=int, default=224 ,          
                        help='size for randomly cropping images')
         
    parser.add_argument('--model_path', type=str, default='./models/adaptive_model-8-1000.pkl',
                        help='path for trained model')
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
