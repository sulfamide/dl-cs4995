import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

        #with attention should be
        #self.linear = nn.Linear(resnet.fc.in_features, 2*embed_size)
        #self.bn = nn.BatchNorm1d(2*embed_size, momentum=0.01)

        self.init_weights()
        
    def init_weights(self):
        """Initialize the weights."""
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
        
    def forward(self, images):
        """Extract the image feature vectors."""
        #print self.linear
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()


    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        #print embeddings.size()
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        print hiddens[0].size(), 'hidden'
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features,max_sent_length,states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        #print features.size(),'decoder feature'

        inputs = features.unsqueeze(1)
        #print inputs.size(),'decoder input'

        for i in range(max_sent_length):                                      # maximum sampling length
            hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size),
            outputs = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)                         # (batch_size, 1, embed_size)
        sampled_ids = torch.cat(sampled_ids, 0)                  # (batch_size, 20)
        return sampled_ids.squeeze()
        #return sampled_ids



class EncoderCNNATT(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNNATT, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-3]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        #self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        #self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weights()

    def init_weights(self):
        """Initialize the weights."""
        pass

    def forward(self, images):
        """Extract the image feature vectors."""

        #model_att = list(self.resnet.children())[:-2]
        #resnet_att = nn.Sequential(*model_att)
        #feat_att = resnet_att(images)
        #print feat_att.size()
        feat_att=self.resnet(images)
        print feat_att.volatile,'feav'
        #print feat_att.size(),'feat_att for'
        return feat_att


class DecoderRNNATT(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNNATT, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm_att = nn.LSTM(1280, 512, 1, batch_first=True)
        self.linear_h0 = nn.Linear(1024, 512)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.att_conv = nn.Conv2d(1536, 1, 1)
        self.att_softmax = nn.Softmax()
        self.init_weights()


    def forward(self,features,captions,length,states=None):
        # features (128L, 1024L, 16L, 16L)
        features=to_var(features.data)
        feat = features.sum(2).sum(2)/256
        #print(feat.size(),'feat')
        #inputs = inputs.unsqueeze(1)
        feat=self.linear_h0(feat)
        #print(feat.size(),'input')
        hiddens=feat.unsqueeze(1)
        inputs=self.embed(to_var(torch.LongTensor(captions.size(0)*[1])))
        outputs=hiddens.squeeze()
        states = (to_var(hiddens.data),to_var(hiddens.data))
        # (batch_size, 1, hidden_size) hidden_size 128
        for i in range(1,captions.size(1)):                                      # maximum sampling length
            att = self.calculateAttension(features, hiddens)
            att = to_var(att.data)
            #print(features.view(128,1024,256).transpose(1,2).size(),'fff')
            #print(att.unsqueeze(1).size(),'att')

            feat_t=features.view(captions.size(0),1024,256).transpose(1,2)
            att_t=att.unsqueeze(1)  #scores    128*1*256
            #print att_t.volatile, 'scv'
            scored_feat=torch.bmm(att_t,feat_t)
            #print scored_feat.volatile,'scv'
            scored_feat=scored_feat.squeeze()
            #print(scored_feat.size(),'scored_feat') # 128 1024

            inputs = torch.cat((inputs,scored_feat),1)
            inputs = inputs.unsqueeze(1)
            #print(inputs.size(),'input')    # 128 1 1280

            hiddens, states = self.lstm_att(inputs, states)          # (batch_size, 1, hidden_size),
            #inputs=  embed
            outputs = torch.cat((outputs,hiddens.squeeze()),0)
            inputs = self.embed(captions[:,i])
            print inputs.volatile, 'inputv'

        print outputs.volatile,'ov'
        outputs = self.linear(outputs)
        outputs = self.unpack_hidden(outputs,length)
        print(outputs.size(), 'output after unpack')   #   2000 * 9956
        return outputs

    def calculateAttension(self, features, hiddens):
        f_size = features.size()
        print features.volatile,'cfv'
        print hiddens.volatile,'chv'
        stacked_feature = hiddens.squeeze();
        # print stacked_feature.size(), 'stacked_size0'
        stacked_feature = stacked_feature.expand(f_size[-1] * f_size[-2], hiddens.size(0), hiddens.size(2))
        # print stacked_feature.size(), 'stacked_size1'
        stacked_feature = stacked_feature.transpose(0, 1).transpose(1, 2)
        # print stacked_feature.size(), 'stacked_size2'
        stacked_feature = stacked_feature.contiguous().view(hiddens.size(0), hiddens.size(2), f_size[-1], f_size[-2])
        # print stacked_feature.size(), 'stacked_size3'

        # (128L, 512L, 16L, 16L) stackedFeature
        # (128L, 1024L, 16L, 16L) features
        att_feature = torch.cat((stacked_feature, features), 1)
        #print att_feature.size(), 'att_feature'

        att_score = self.att_conv(att_feature).squeeze()
        att_score = self.att_softmax(att_score.view(att_score.size(0), -1))
        #print att_score.size()

        pass
        return att_score

    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.linear_h0.weight.data.uniform_(-0.1, 0.1)
        self.linear_h0.bias.data.fill_(0)
        self.att_conv.weight.data.normal_(0.0, 0.02)
        self.att_conv.bias.data.fill_(0)

    def unpack_hidden(self,data,batch_sizes):
        max_len=batch_sizes[0]
        output=data[0:batch_sizes[0]]
        for i in range(1,len(batch_sizes)):
            output = torch.cat((output,data[i*max_len:i*max_len+batch_sizes[i],:]),0)
        return output

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)
