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


class EncoderRNN(nn.Module):
    def __init__(self, gate, embeddings, hidden_size, gaussian_dim, num_gaussians, num_layers,
                 dropout, fixed_embeddings=False, bidirectional=False, general_covariance=False):
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
        """
        super(EncoderRNN, self).__init__()
        self.num_gaussians = num_gaussians
        self.gaussian_dim = gaussian_dim
        self.gate = gate
        self.bidirectional = bidirectional
        self.general_covariance = general_covariance
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

        # Feed-Forward from rnn_dim to output_dim: mu + sigma
        avg_size = self.gaussian_dim  # mu~(dim)
        var_size = self.gaussian_dim if not self.general_covariance else self.gaussian_dim ** 2  # variance~(dim)
        if self.bidirectional:
            self.fc_avg = nn.Linear(hidden_size * 2, avg_size * self.num_gaussians)
            self.fc_var = nn.Linear(hidden_size * 2, var_size * self.num_gaussians)
        else:
            self.fc_avg = nn.Linear(hidden_size, avg_size * self.num_gaussians)
            self.fc_var = nn.Linear(hidden_size, var_size * self.num_gaussians)
        self.init_weights()

    def init_weights(self):
        self.fc_avg.weight.data.uniform_(-0.1, 0.1)
        self.fc_avg.bias.data.fill_(0)
        self.fc_var.weight.data.uniform_(-0.1, 0.1)
        self.fc_var.bias.data.fill_(0)

    def _reparametrize(self, mus, var):
        # build covariance matrix from list of variances, ie:
        #       [   var(x_1)   | cov(x_1 x_2) | cov(x_1 x_3) ]
        # cov = [ cov(x_2 x_1) |   var(x_2)   | cov(x_2 x_3) ]
        #       [ cov(x_3 x_1) | cov(x_3 x_2) |   var(x_3)   ]
        '''
        TODO: do the following using ONLY torch!

        np_mus = mus.data.numpy()
        np_var = log_var.data.numpy()
        batch_size = np_mus.shape[0]
        np_z = numpy.zeros((batch_size, self.num_gaussians, self.gaussian_dim))

        # for each gaussian
        for idx in range(0, self.num_gaussians*self.gaussian_dim, self.num_gaussians):
            # meanS for each example
            np_mu = np_mus[:, idx:idx+self.gaussian_dim]  # ~(bs, dim)
            # covarianceS for each example
            np_cov = np.array([  # covariance = outer product between variances
                np.outer(
                    np_var[ex, idx:idx+self.gaussian_dim],
                    np_var[ex, idx:idx+self.gaussian_dim]
                ) for ex in range(batch_size)
            ])  # ~(bs, dim, dim)

            # sample for each example according to its mixture of gaussian
            samples = np.array([
                np.random.multivariate_normal(
                    np_mu[ex], np_cov[ex]
                ) for ex in range(batch_size)
            ])  # ~(bs, dim)

            # populate z tensor
            np_z[:, idx/self.gaussian_dim, :] = samples

        np_z = np_z.reshape(batch_size, -1)  # reshape to (bs, num_gaussians * gaussian_dim)
        return torch.from_numpy(np_z).float()
        '''

        # assume mus shape is (bs, num_gaussians * gaussian_dim)
        # assume vars shape is (bs, num_gaussians * gaussian_dim if diagonal else num_gaussians * gaussian_dim ** 2)

        eps = to_var(torch.randn(mus.size(0), mus.size(1)))  # ~(bs, num_gaussians * mu_size=dim)
        if not self.general_covariance:
            z = mus + eps * torch.exp(var / 2)
        else:
            var = var.view((-1, self.num_gaussians, self.gaussian_dim, self.gaussian_dim))
            eps = eps.view((-1, self.num_gaussians, self.gaussian_dim, 1))
            mus = mus.view((-1, self.num_gaussians, self.gaussian_dim))
            z = []
            for i in range(self.num_gaussians):
                z.append(mus[:, i, :] + torch.bmm(torch.exp(var[:, i, :, :] / 2), eps[:, i, :]).view(-1, self.gaussian_dim))
            z = torch.cat(z, dim=-1)
        return z

    def forward(self, captions, lengths, sample=True):
        embeddings = self.embed(captions)  # ~(bs, max_len, embed)
        print("embeddings: %s" % (embeddings.size(),))

        # pack sequences to avoid calculation on padded elements
        # understood from: https://discuss.pytorch.org/t/understanding-pack-padded-sequence-and-pad-packed-sequence/4099/5
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)  # makes a PackedSequence

        # output ~(batch, seq_length, hidden_size * num_directions) <-- output features (h_t) from the last layer of the RNN, for each t.
        # hidden ~(num_layers * num_directions, batch, hidden_size) <-- tensor containing the hidden state for t=seq_len
        output, hidden = self.rnn(packed)  # encode caption
        output, lengths_ = pad_packed_sequence(output, batch_first=True)  # convert back to Tensor
        assert lengths == lengths_
        # hiddens, out = self.rnn(packed)  # encode caption out~(num_layers*num_directions, batch, hidden_size)
        print('Output size:', output.size(), '- Hidden sizes:', [h.size() for h in hidden])

        # lstm returns hidden state AND cell state
        if self.gate == 'lstm':
            hidden, cell = hidden

        # unidirectional
        # gru / rnn - 1 layer  --> take hidden[-1] bcs only one                          or   output[:, -1, :]
        # lstm      - 1 layer  --> take hidden[0][-1] bcs [1][..] is only the cell state or   output[:, -1, :]
        # gru / rnn - k layers --> take hidden[-1] bcs other are previous layers         or   output[:, -1, :]
        # lstm      - k layers --> take hidden[0][-1] bcs [0][..] are previsou layers    or   output[:, -1, :]
        # bidirectional
        # gru / rnn - 1 layer  --> concat hidden[0 & 1]                                       or   output[:, -1, :]
        # lstm      - 1 layer  --> concat hidden[0][0 & 1] bcs [1][..] is only the cell state or   output[:, -1, :]
        # gru / rnn - k layers --> concat hidden[?? & ??] bcs returning 2*k states            or   output[:, -1, :]
        # lstm      - k layers --> take hidden[0][?? & ??] bcs returning 2*k states           or   output[:, -1, :]

        mus = self.fc_avg(output[:, -1, :])  # map to gaussian means
        sds = self.fc_var(output[:, -1, :])  # map to gaussian variances

        return self._reparametrize(mus, sds), mus[0], sds[0]


# TODO: Add Deconvolutions (replace fully connected (fc) layers)
# see: https://github.com/SherlockLiao/pytorch-beginner/tree/master/08-AutoEncoder
# see: http://pytorch.org/docs/master/nn.html#torch.nn.ConvTranspose2d
class DecoderCNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DecoderCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], output_size)

    def init_weights(self):
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc1.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-0.1, 0.1)
        self.fc2.bias.data.fill_(0)

    def forward(self, x):
        o = F.relu(self.fc1(x))
        o = self.fc2(o)
        return o
