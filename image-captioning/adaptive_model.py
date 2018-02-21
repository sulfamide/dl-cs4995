import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import init
import torch.nn.functional as func
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


class EncoderCNN(nn.Module):
    def __init__(self, embed_size,hidden_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-2]          # delete the last fc layer and the avgpool layer .
        self.resnet = nn.Sequential(*modules)
        self.avgpool = nn.AvgPool2d(7)
        self.affine_a = nn.Linear(2048, hidden_size)    # v_i = W_a * A
        self.affine_b = nn.Linear(2048, embed_size)     # v_g = W_b * a^g

        # Dropout before affine transformation
        self.dropout = nn.Dropout(0.5)

        self.init_weights()


    def init_weights(self):
        """Initialize the weights."""
        init.kaiming_uniform(self.affine_a.weight, mode='fan_in')
        init.kaiming_uniform(self.affine_b.weight, mode='fan_in')
        self.affine_a.bias.data.fill_(0)
        self.affine_b.bias.data.fill_(0)


    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.resnet(images)
        features = to_var(features.data)
        avg_features = self.avgpool(features)
        avg_features = avg_features.view(avg_features.size(0),-1)

        print(features.size())
        V = features.view(features.size(0),features.size(1),-1).transpose(1,2)
        V = func.relu(self.affine_a(self.dropout(V)))                           # bsz * res * hidden   res: size(2)*size(3) of output of resnet

        v_g=func.relu(self.affine_b(self.dropout( avg_features)))               # bsz * emb

        return V,v_g

class Attention(nn.Module):
    def __init__(self,hidden_size):
        super(Attention, self).__init__()

        self.affine_v = nn.Linear(hidden_size,49,bias=False)
        self.affine_g = nn.Linear(hidden_size,49,bias=False)
        self.affine_s = nn.Linear(hidden_size,49,bias=False)
        self.affine_h = nn.Linear(49,1,bias=False)

        self.dropout = nn.Dropout(0.5)
        self.init_weights()


    def init_weights(self):
        init.xavier_uniform(self.affine_v.weight)
        init.xavier_uniform(self.affine_g.weight)
        init.xavier_uniform(self.affine_h.weight)
        init.xavier_uniform(self.affine_s.weight)

    def forward(self,V,h_t,sent_t):
        # V : bsz * res * hidden
        # sent_t : bsz * seq * hidden
        # h_t : bsz * seq * hidden

        content_v = self.affine_v(self.dropout(V)).unsqueeze(1)+self.affine_g(self.dropout(h_t)).unsqueeze(2)
        # bsz * 1 * res * 49 + bsz * seq * 1 * 49 = bsz * seq * res * 49

        z_t = self.affine_h(self.dropout(func.tanh(content_v))).squeeze(3)                   # bsz * seq * res
        alpha_t = func.softmax(z_t.view(-1,z_t.size(2))).view(z_t.size(0),z_t.size(1),-1)   # bsz * seq * res

        c_t = torch.bmm(alpha_t,V)  # bsz * seq * hidden

        content_s = self.affine_s(self.dropout(sent_t)) + self.affine_g(self.dropout(h_t))  # bsz * seq * 49
        z_t_extend = self.affine_h(self.dropout(func.tanh(content_s)))                      # bsz * seq * 1
        extend = torch.cat((z_t,z_t_extend),2)                                              # bsz * seq * 1+res

        alpha_hat_t = func.softmax(extend.view(-1,extend.size(2))).view(extend.size(0),extend.size(1),-1)    # bsz * seq * 1+res
        beta_t = alpha_hat_t[:,:,-1]     # bsz * seq
        beta_t = beta_t.unsqueeze(2)     # bsz * seq * 1

        c_hat_t = beta_t * sent_t + (1-beta_t) * c_t  # bsz * seq * hidden

        return c_hat_t,alpha_t,beta_t


class Sentinel(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(Sentinel, self).__init__()

        self.affine_x = nn.Linear(input_size,hidden_size,bias=False)
        self.affine_h = nn.Linear(hidden_size,hidden_size,bias=False)

        self.dropout = nn.Dropout(0.5)
        self.init_weights()


    def init_weights(self):
        init.xavier_uniform(self.affine_h.weight)
        init.xavier_uniform(self.affine_h.weight)

    def forward(self,x,hiddens_t,cells):
        # x : bsz * seq * 2emb
        # hiddens : bsz * seq * hidden
        # cells : bsz * seq * hidden

        gate_t = self.affine_x(self.dropout(x)) + self.affine_h(self.dropout(hiddens_t))
        gate_t = func.sigmoid(gate_t) # bsz * seq * hidden

        sent_t = gate_t * func.tanh(cells) # bsz * seq * hidden
        return sent_t

class Adaptive(nn.Module):
    def __init__(self,embed_size,hidden_size,vocab_size):
        super(Adaptive, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.sentinel = Sentinel(2*embed_size,hidden_size)
        self.attention = Attention(hidden_size)

        self.mlp = nn.Linear(hidden_size,vocab_size)
        self.init_weights()

    def init_weights(self):
        init.kaiming_normal(self.mlp.weight, mode='fan_in')
        self.mlp.bias.data.fill_(0)

    def forward(self,x,hiddens,cells,V):

        h_0=self.init_hidden(hiddens.size(0))[0].transpose(0,1)
        if hiddens.size(1)>1:
            hiddens_t_1 = torch.cat((h_0,hiddens[:,:-1,:]),1)   # bsz * seq * hidden
        else:
            hiddens_t_1 = h_0

        sentinel = self.sentinel(x,hiddens_t_1,cells)

        c_hat , att_scores, beta = self.attention(V,hiddens,sentinel)
        # c_hat: bsz * seq * hidden

        output = self.mlp(self.dropout(c_hat + hiddens))  # bsz * seq * vocab_size

        return output,att_scores,beta

    def init_hidden(self,bsz):
        weight = next(self.parameters()).data
        if torch.cuda.is_available():
            return (Variable(weight.new(1, bsz, self.hidden_size).zero_().cuda()),
                    Variable(weight.new(1, bsz, self.hidden_size).zero_().cuda()))
        else:
            return (Variable(weight.new(1, bsz, self.hidden_size).zero_()),
                    Variable(weight.new(1, bsz, self.hidden_size).zero_()))


class DecoderRNN(nn.Module):
    def __init__(self,embed_size,vocab_size,hidden_size):
        super(DecoderRNN, self).__init__()

        self.embed = nn.Embedding(vocab_size,embed_size)
        self.lstm = nn.LSTM(2*embed_size,hidden_size,1,batch_first=True)
        self.hidden_size = hidden_size
        self.adaptive= Adaptive(embed_size,hidden_size,vocab_size)

    def forward(self,V,v_g,captions,states=None):

        embeddings = self.embed(captions)

        x = torch.cat((embeddings, v_g.unsqueeze(1).expand_as(embeddings)),dim=2)  # x=[embed,v] bsz * seq * 2emb

        if torch.cuda.is_available():
            hiddens = Variable(torch.zeros(x.size(0), x.size(1), self.hidden_size).cuda())  #bsz * seq * hidden
            cells = Variable(torch.zeros(x.size(1), x.size(0), self.hidden_size).cuda())   #seq * bsz * hidden
        else:
            hiddens = Variable(torch.zeros(x.size(0), x.size(1), self.hidden_size))
            cells = Variable(torch.zeros(x.size(1), x.size(0), self.hidden_size))

        for time_step in range(x.size(1)):
            x_t = x[:,time_step,:]
            x_t = x_t.unsqueeze(1)   # bsz * 1 * 2emd

            h_t,states =self.lstm(x_t,states)

            hiddens[:,time_step,:] =  h_t
            cells[time_step,:,:] = states[1]

        cells = cells.transpose(0,1)  # bsz * seq * hidden

        output, att_score, beta =self.adaptive(x,hiddens,cells,V)

        return output,att_score,beta,states

class Encoder2Decoder(nn.Module):
    def __init__(self,embed_size,vocab_size,hidden_size):
        super(Encoder2Decoder, self).__init__()

        # encoder and decoder
        self.encoder=EncoderCNN(embed_size,hidden_size)
        self.decoder=DecoderRNN(embed_size,vocab_size,hidden_size)

    def forward(self,images,captions,lengths):

        V,v_g = self.encoder(images)
        outputs, _, _, _ =self.decoder(V,v_g,captions)
        packed_outputs = pack_padded_sequence(outputs, lengths, batch_first=True)
        return packed_outputs


    def sample(self,images,max_length):
        V,v_g = self.encoder(images)

        if torch.cuda.is_available():
            captions = Variable(torch.LongTensor(images.size(0), 1).fill_(1).cuda())
        else:
            captions = Variable(torch.LongTensor(images.size(0), 1).fill_(1))

        sampled_ids=[]
        attention=[]
        Beta=[]

        states = None

        for i in range(max_length):
            scores,att_scores,beta,states = self.decoder(V,v_g,captions,states)
            predicted = scores.max(2)[1]
            captions = predicted
            sampled_ids.append(captions)
            attention.append(att_scores)
            Beta.append(beta)

        sampled_ids = torch.cat(sampled_ids,1)
        attention = torch.cat(attention,1)
        Beta = torch.cat(beta,1)


        return sampled_ids,attention,Beta