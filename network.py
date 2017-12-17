import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def softmax(input, axis):
    """
    taken from https://discuss.pytorch.org/t/why-softmax-function-cant-specify-the-dimension-to-operate/2637
    """
    input_size = input.size()
    trans_input = input.transpose(axis, -1)
    trans_size = trans_input.size()
    input_2d = trans_input.contiguous().view(-1, trans_size[-1])
    soft_max_2d = F.softmax(input_2d)
    soft_max_nd = soft_max_2d.view(*trans_size)
    return soft_max_nd.transpose(axis, -1)


class EncoderRNN(nn.Module):
    def __init__(self, gate, embeddings, hidden_size, gaussian_dim, num_gaussians, num_layers,
                 dropout, fixed_embeddings=False, bidirectional=False, indep_gaussians=True):
        """
        Encoder network that works on the word level of captions.
        :param gate: one of 'rnn', 'gru', or 'lstm'
        :param embeddings: |vocab|x|word_vector| numpy matrix
        :param hidden_size: size of the caption embedding
        :param gaussian_dim: dimmension of each gaussian variable
        :param num_gaussians: number of gaussian variables to sample
        :param num_layers: number of hidden layers for the encoder
        :param dropout: if non-zero, will add a dropout layer to the rnn
        :param fixed_embeddings: freeze word embeddings during training
        :param bidirectional: bidirectional rnn
        :param indep_gaussians: sample from indep gaussians, otherwise will sample from mixture
        """
        super(EncoderRNN, self).__init__()
        self.num_gaussians = num_gaussians
        self.gaussian_dim = gaussian_dim
        self.gate = gate
        self.bidirectional = bidirectional
        self.indep_gaussians = indep_gaussians
        if self.indep_gaussians:
            print('|__sampling from indep gaussian distributions')
        else:
            print('|__sampling from multivariate gaussian distributions')
        # Input: word vector
        embeddings = torch.from_numpy(embeddings).float()  # create a pytorch tensor from numpy array
        self.embed = nn.Embedding(embeddings.size(0), embeddings.size(1))  # create embedding layer
        self.embed.weight = nn.Parameter(embeddings, requires_grad=(not fixed_embeddings))  # set value

        # Encoder: rnn, gru, or lstm
        if self.gate == 'rnn':
            self.rnn = nn.RNN(input_size=self.embed.embedding_dim,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              batch_first=True,  # in & out dim ~(batch, sequence, features)
                              dropout=dropout,
                              bidirectional=self.bidirectional)
        elif self.gate == 'gru':
            self.rnn = nn.GRU(input_size=self.embed.embedding_dim,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              batch_first=True,  # in & out dim ~(batch, sequence, features)
                              dropout=dropout,
                              bidirectional=self.bidirectional)
        elif self.gate == 'lstm':
            self.rnn = nn.LSTM(input_size=self.embed.embedding_dim,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              batch_first=True,  # in & out dim ~(batch, sequence, features)
                              dropout=dropout,
                              bidirectional=self.bidirectional)
        else:
            print("ERROR: unknown encoder gate: %s" % self.gate)
            return

        # Feed-Forward: from rnn_dim to output_dim: mu + sigma
        true_hidden_size = hidden_size*2 if self.bidirectional else hidden_size

        self.att = nn.Linear(true_hidden_size, true_hidden_size)  # attention layer

        self.fc_avg = nn.Linear(true_hidden_size, self.gaussian_dim * self.num_gaussians)
        self.fc_var = nn.Linear(true_hidden_size, self.gaussian_dim * self.num_gaussians)

        self.init_weights()

    def init_weights(self):
        # rnn parameters
        for name, param in self.rnn.named_parameters():
            if name.startswith('weight'):
                param.data.normal_(0., 0.1)
            elif name.startswith('bias'):
                param.data.fill_(0)
            else:
                print('default initialization for parameter %s' % name)
        # fully connected parameters
        self.fc_avg.weight.data.uniform_(-0.1, 0.1)
        self.fc_avg.bias.data.fill_(0)
        self.fc_var.weight.data.uniform_(-0.1, 0.1)
        self.fc_var.bias.data.fill_(0)
        # attention parameters
        self.att.weight.data.uniform_(-0.1, 0.1)
        self.att.bias.data.fill_(0)

    def _reparametrize(self, mus, log_vars):
        # build covariance matrix from list of variances, ie:
        #       [   var(x_1)   | cov(x_1 x_2) | cov(x_1 x_3) ]
        # cov = [ cov(x_2 x_1) |   var(x_2)   | cov(x_2 x_3) ]
        #       [ cov(x_3 x_1) | cov(x_3 x_2) |   var(x_3)   ]

        # assume mus shape is (bs, num_gaussians * gaussian_dim)
        # assume variances shape is (bs, num_gaussians * gaussian_dim)

        eps = to_var(torch.randn(mus.size(0), mus.size(1)))  # ~(bs, num_gaussians * mu_size=dim)
        if self.indep_gaussians:
            z = mus + eps * torch.exp(log_vars / 2)
        else:
            log_vars = log_vars.view((-1, self.num_gaussians, self.gaussian_dim))
            eps = eps.view((-1, self.num_gaussians, self.gaussian_dim, 1))
            mus = mus.view((-1, self.num_gaussians, self.gaussian_dim))
            z = []  # list of samples each of shape ~(bs, gaussian_dim)
            for i in range(self.num_gaussians):
                # covarianceS: batch outer product between exp{variances/2}
                cov = torch.bmm(  # TODO: why do we take `exp{var/2}` and not simply `exp{var}` ???
                        torch.exp(log_vars[:, i, :]/2.).view(-1, self.gaussian_dim, 1),
                        torch.exp(log_vars[:, i, :]/2.).view(-1, 1, self.gaussian_dim)
                )  # ~(bs, dim, dim)

                z.append(mus[:, i, :] + torch.bmm(cov, eps[:, i, :]).view(-1, self.gaussian_dim))
            z = torch.cat(z, dim=-1)
        return z


    def forward(self, captions, lengths, sample=True):
        embeddings = self.embed(captions)  # ~(bs, max_len, embed)

        # pack sequences to avoid calculation on padded elements
        # see: https://discuss.pytorch.org/t/understanding-pack-padded-sequence-and-pad-packed-sequence/4099/5
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)  # convert embeddings to PackedSequence

        # output ~(batch, seq_length, hidden_size*num_directions) <-- output features (h_t) from the last layer of the RNN, for each t.
        # hidden ~(num_layers*num_directions, batch, hidden_size) <-- tensor containing the hidden state for t=seq_len
        output, hidden = self.rnn(packed)  # encode caption
        # lstm returns hidden state AND cell state
        if self.gate == 'lstm':
            hidden, cell = hidden

        # convert back output PackedSequence to tensor
        output, lengths_ = pad_packed_sequence(output, batch_first=True)
        assert lengths == lengths_

        # print('Output size:', output.size(), '- Hidden sizes:', [h.size() for h in hidden])

        # TODO: compute attention energies
        # energies = self.att(output.view(-1, output.size(2))).view(output.size())  # ~ (bs, seq_len, enc)
        # energies = energies[:, :lengths, :]  # TODO: remove unused energies for short sequences
        # alpha = softmax(energies, 1)  # apply softmax on dim=1

        # grab the encodding of the sentence, not the padded part!
        encoded = output[
                range(output.size(0)),  # take each sentence
                list(map(lambda l: l-1, lengths)),  # at their last index (ie: length-1)
                :  # take full encodding
        ]  # ~ (bs, enc)

        mus = self.fc_avg(encoded)  # map to gaussian means
        sds = self.fc_var(encoded)  # map to gaussian variances
        z = self._reparametrize(mus, sds)

        return z, mus, sds


# TODO: Add Deconvolutions (replace fully connected (fc) layers)
# see: https://github.com/SherlockLiao/pytorch-beginner/tree/master/08-AutoEncoder
# see: http://pytorch.org/docs/master/nn.html#torch.nn.ConvTranspose2d
class DecoderCNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate):
        """
        :type input_size: int
        :type hidden_sizes: list of ints for each hidden layers
        :type output_size: int
        """
        super(DecoderCNN, self).__init__()

        self.hidden_sizes = hidden_sizes
        self.layers = []

        # unfortunately PyTorch hardly supports definition of hidden layers in a loop.
        # see: https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219
        # dumb solution: check list length and create layers incrementaly...
        self.fc_1 = nn.Linear(input_size, hidden_sizes[0])
        self.layers.append(self.fc_1)
        if len(self.hidden_sizes) >= 2:
            self.fc_2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
            self.layers.append(self.fc_2)
        if len(self.hidden_sizes) >= 3:
            self.fc_3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
            self.layers.append(self.fc_3)
        if len(self.hidden_sizes) >= 4:
            self.fc_4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
            self.layers.append(self.fc_4)
        if len(self.hidden_sizes) >= 5:
            self.fc_5 = nn.Linear(hidden_sizes[3], hidden_sizes[4])
            self.layers.append(self.fc_5)
        if len(self.hidden_sizes) >= 6:
            self.fc_6 = nn.Linear(hidden_sizes[4], hidden_sizes[5])
            self.layers.append(self.fc_6)
        if len(self.hidden_sizes) > 6:
            print('WARNING: layer %s will not be considered. Update code')
            self.fc_last = nn.Linear(hidden_sizes[5], output_size)
        else:
            self.fc_last = nn.Linear(hidden_sizes[-1], output_size)
        self.layers.append(self.fc_last)

        self.init_weights()

        self.dropout = nn.Dropout(p=dropout_rate)

    def init_weights(self):
        """
        initialize all weights for all decoder layers
        """
        for layer in self.layers:
            layer.weight.data.uniform_(-0.1, 0.1)
            layer.bias.data.fill_(0)

    def forward(self, x):
        o = x  # start with output = input
        # for all hidden layers except the last one
        for layer in self.layers[:-1]:
            o = F.relu(layer(o))  # apply relu
        o = self.dropout(o)
        o = self.layers[-1](o)    # last layer: no RELU
        return o

