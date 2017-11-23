import time
import pickle
import argparse
import numpy as np
from gensim.models import KeyedVectors
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms

from network import EncoderRNN, DecoderCNN
from data_loader import get_loader
from build_vocab import Vocabulary


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def load_glove_vec(fname):
    """
    Loads word vecs from gloVe
    """
    word_vecs = {}
    length = 0
    with open(fname, "rb") as f:
        for i, line in enumerate(f):
            L = line.split()
            word = L[0].lower()
            word_vecs[word] = np.array(L[1:], dtype='float32')
            if length == 0:
                length = len(word_vecs[word])
    return word_vecs, length


def run(args):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    print("Loading vocab...")
    with open(args.vocab_loc, 'rb') as f:
        vocab = pickle.load(f)
    print("number of unique tokens: %d" % len(vocab))

    print("Get data loader...")
    train_loader = get_loader(root=args.images_loc,
                              json=args.captions_loc,
                              vocab=vocab, batch_size=args.batch_size, num_workers=2, shuffle=True, transform=transform)

    # Input: word vector
    if args.embeddings_loc:
        print("\nLoading word embeddings from %s" % args.embeddings_loc)

        if 'google' in args.embeddings_loc.lower() and args.embeddings_loc.endswith('.bin'):
            w2v = KeyedVectors.load_word2vec_format(args.embeddings_loc, binary=True)
            emb_size = w2v.vector_size
        elif 'glove' in args.embeddings_loc.lower() and args.embeddings_loc.endswith('.txt'):
            w2v, emb_size = load_glove_vec(args.embeddings_loc)
        else:
            print("ERROR: unknown embedding file %s" % args.embeddings_loc)
            return

        embeddings = np.random.uniform(-0.1, 0.1, size=(len(vocab), emb_size))
        for word, idx in vocab.word2idx.items():
            if word in w2v:
                embeddings[idx] = w2v[word]
    else:
        print("\nCreating random word embeddings of size %dx%d" % (len(vocab), args.embedding_size))
        embeddings = np.random.uniform(-0.1, 0.1, size=(len(vocab), args.embedding_size))

    print("\nCreating encoder...")
    encoder = EncoderRNN(args.encoder_gate, embeddings, args.encoding_size, args.gaussian_dim, args.num_gaussians,
                         args.encoder_layers, args.dropout_rate, fixed_embeddings=args.fix_embeddings, bidirectional=args.bidirectional)
    print("Creating decoder...")
    decoder = DecoderCNN(input_size=args.num_gaussians*args.gaussian_dim,
                         output_size=train_loader.dataset[0][0].view(-1).size(0))

    if torch.cuda.is_available():
        print("\ncuda available!")
        print("Moving variables to cuda %d..." % torch.cuda.current_device())
        encoder.cuda()
        decoder.cuda()

    # fetch parameters to train
    params = list(encoder.parameters()) + list(decoder.parameters())
    params = filter(lambda p: p.requires_grad, params)  # if we decide to fix parameters, ignore them
    # create optimizer ('adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta')
    if args.optimizer == 'adam':
        optimizer = optim.Adam(params=params, lr=args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(params=params, lr=args.learning_rate)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(params=params, lr=args.learning_rate)
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(params=params, lr=args.learning_rate)
    elif args.optimizer == 'adadelta':
        optimizer = optim.Adadelta(params=params, lr=args.learning_rate)
    else:
        print("ERROR: unknown optimizer: %s" % args.optimizer)
        return

    # TODO add KL with covarariance matrix
    def kl(mu, log_var):
        return (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())) / mu.view(-1).size(0)

    recon = torch.nn.MSELoss()

    start_time = time.time()
    print("\nTraining model...")
    for epoch in range(args.epochs):
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        nb_train_batches = 0
        for i, (images, captions, lengths) in enumerate(train_loader):
            optimizer.zero_grad()

            images = to_var(images)
            captions = to_var(captions)

            sampled, mu, log_var = encoder(captions, lengths)
            outputs = decoder(sampled)

            kl_loss = kl(mu, log_var)
            recon_loss = recon(outputs, images)
            print("epoch %.2d - step %.3d - kl loss %.6f - recon loss %.6f" % (epoch+1, i+1, round(kl_loss.data[0], 6), round(recon_loss.data[0], 6)))

            loss = kl_loss + 2*recon_loss  # put more emphasis on the reconstruction loss?
            loss.backward()
            optimizer.step()
            epoch_recon_loss += recon_loss.data[0]
            epoch_kl_loss += kl_loss.data[0]
            nb_train_batches += 1

            # in debug mode, break the training loop after 10 batches
            if args.debug and nb_train_batches == 10:
                break

        epoch_recon_loss /= nb_train_batches
        epoch_kl_loss /= nb_train_batches
        print("Epoch: ", epoch, "recon loss: ", round(epoch_recon_loss, 4),
              "kl loss: ", round(epoch_kl_loss, 4))

    print('Finished Training, time elapsed: ', round(time.time() - start_time, 2), ' seconds')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', '-d', action='store_true', help="stop training after 10 batches")
    # training params
    parser.add_argument('--epochs',        '-e' , type=int, default=1, help="num epochs")
    parser.add_argument('--batch_size',    '-bs', type=int, default=32, help="mini batch size")
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--optimizer',     '-opt',choices=['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta'], default='adam', help="optimizer to use")
    # network architecture
    parser.add_argument('--num_gaussians', '-ng', type=int, default=2, help="num gaussians")
    parser.add_argument('--gaussian_dim',  '-gd', type=int, default=16, help="dimension of each gaussian variable")
    parser.add_argument('--activation',    '-ac', choices=['sigmoid', 'relu', 'swish'], type=str, default='relu', help="activation function")
    parser.add_argument('--dropout_rate',  '-dr', type=float, default=0.0, help="probability of dropout layer")
    ## encoder network
    parser.add_argument('--encoder_gate', choices=['rnn', 'gru', 'lstm'], default='rnn', help="recurrent network gate")
    parser.add_argument('--bidirectional',  type=str2bool, default='False', help="bidirectional encoder")
    parser.add_argument('--embedding_size', type=int, default=300, help="size of word vectors")
    parser.add_argument('--fix_embeddings', type=str2bool, default='False', help="don't train word embeddings")
    parser.add_argument('--encoding_size',  type=int, default=500, help="size of caption vectors")
    parser.add_argument('--encoder_layers', type=int, default=1, help="number of hidden layers in the caption encoder")
    ## decoder network
    parser.add_argument('--decoder_layers', nargs='+', type=int, default=[256], help="List of hidden sizes for the de-convolution network")
    # data files
    parser.add_argument('--images_loc', '-il', default='/coco/images/resized2014', help="location of resized images")
    parser.add_argument('--captions_loc', '-cl', default='/coco/annotations/captions_train2014.json', help="location of captions")
    parser.add_argument('--vocab_loc', '-vl', default='./data/vocab.pkl', help="location of vocabulary")
    parser.add_argument('--embeddings_loc', '-el', default=None, help="location of pretrained word embeddings")
    args = parser.parse_args()
    print('args: %s\n' % args)

    run(args)


