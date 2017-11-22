import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


#  TODO : Add Covariance Matrix
#  TODO : Replace nn.Embedding with word2vec
class EncoderRNN(nn.Module):
    def __init__(self, embeddings, hidden_size, gaussian_dim, num_gaussians, num_layers,
                 fixed_embeddings=False):
        """
        Encoder network that works on the word level of captions.
        :param embeddings: |vocab|x|word_vector| numpy matrix
        :param hidden_size: size of the caption embedding
        :param gaussian_dim: dimmension of each gaussian variable
        :param num_gaussians: number of gaussian variables to sample
        :param num_layers: number of hidden layers for the encoder
        :param fixed_embeddings: freeze word embeddings during training
        """
        super(EncoderRNN, self).__init__()
        self.num_gaussians = num_gaussians
        self.gaussian_dim = gaussian_dim
        # Input: word vector
        embeddings = torch.from_numpy(embeddings)  # create a pytorch tensor from numpy array
        embeddings = embeddings.type(torch.FloatTensor)  # cast to FloatTensor
        self.embed = nn.Embedding(embeddings.size(0), embeddings.size(1))  # create embedding layer
        self.embed.weight = nn.Parameter(embeddings, requires_grad=(not fixed_embeddings))  # set value

        # TODO: init with word2vec
        # RNN | GRU | LSTM ?
        self.rnn = nn.RNN(input_size=self.embed.embedding_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # Feed-Forward from rnn_dim to output_dim
        gaussian_size = self.gaussian_dim + self.gaussian_dim  # 1 gaussian = mu~(dim) + var~(dim)
        self.fc = nn.Linear(hidden_size, gaussian_size * self.num_gaussians)
        self.init_weights()

    def init_weights(self):
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)

    # TODO? handle mixture of gaussians rather than list of indep gaussians? ie: covariance matrix instead of log_var?
    def _reparametrize(self, mu, log_var):
        eps = to_var(torch.randn(mu.size(0), mu.size(1)))  # ~(bs, k*dim)
        z = mu + eps * torch.exp(log_var / 2)
        return z

    def forward(self, captions, lengths, sample=True):
        embeddings = self.embed(captions)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)  # not 100% sure about this
        hiddens, out = self.rnn(packed)

        mus, log_vars = torch.chunk(self.fc(out[0]), 2, dim=-1)  # mus & log_vars ~(bs, k*dim)
        return self._reparametrize(mus, log_vars), mus[0], log_vars[0]


#  TODO: Add Deconvolutions (replace fully connected (fc) layers
class DecoderCNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DecoderCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, output_size)

    def init_weights(self):
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc1.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-0.1, 0.1)
        self.fc2.bias.data.fill_(0)

    def forward(self, x):
        o = F.relu(self.fc1(x))
        o = self.fc2(o)
        return o
