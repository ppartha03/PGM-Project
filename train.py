import os
import time
import pickle
import argparse

from PIL import Image
import numpy as np
from gensim.models import KeyedVectors
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

from network import EncoderRNN, DecoderCNN
from data_loader import get_loader, get_mnist_loader
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


# KL loss  # TODO: why not using torch.nn.KLDivLoss ??
def kl(mu, log_var):
    # if args.indep_gaussians:
    #     # TODO: paste source url for this... see another definition: https://stats.stackexchange.com/a/281725
    return (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())) / mu.view(-1).size(0)
    # The above seems to work fine with mixture of gaussians as well!
    '''
    else:
        # TODO: DIVERGING!! :( :(
        mu = mu.view((-1, args.num_gaussians, args.gaussian_dim))
        log_var = log_var.view((-1, args.num_gaussians, args.gaussian_dim))
        total_kl = 0
        for g in range(args.num_gaussians):
            for b in range(log_var.size(0)):
                log_var_g_b = log_var[b, g, :]  # ~(dim)
                cov_g_b = torch.ger(
                        torch.exp(log_var_g_b),
                        torch.exp(log_var_g_b)
                )  # ~(dim x dim)
                mu_g_b = mu[b, g, :]  # ~(dim)
                # TODO: paste source url for this... see another definition: https://stats.stackexchange.com/a/281725
                total_kl += 0.5 * torch.sum(
                        args.gaussian_dim + cov_g_b - torch.trace(cov_g_b) - mu_g_b**2
                )
        total_kl /= mu.view(-1).size(0)
        return total_kl
    '''


def load_data(args):
    """
    Load vocabulary, data_loaders, word embeddings
    """
    if args.use_mnist:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,),
                                 (0.3081,))
        ])
        print("Loading vocab...")
        with open(args.vocab_loc, 'rb') as f:
            vocab = pickle.load(f)
        print("number of unique tokens: %d" % len(vocab))

        print("Get data loader...")
        train_loader = get_mnist_loader(
                vocab=vocab, train=True, download=True,
                transform=transform,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=2
        )
        test_loader = get_mnist_loader(
                vocab=vocab, train=False, download=True,
                transform=transform,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=2

        )

    else:
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
        train_loader = get_loader(
                root=args.images_loc, json=args.captions_loc, vocab=vocab, train=True,
                transform=transform,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=2
        )
        test_loader = get_loader(
                root=args.images_loc, json=args.captions_loc, vocab=vocab, train=False,
                transform=transform,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=2
        )

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

    return vocab, train_loader, test_loader, embeddings


def show_images(captions, images, outputs, use_mnist, vocab, k=1, prefix='generated/'):
    """
    show k instances: caption, true image, reconstruction
    if mnist: 1 x  28 x  28
    if coco : 3 x 256 x 256
    """
    assert len(captions) == len(images) == len(outputs)
    idx = np.random.choice(range(len(images)), k, replace=False)  # sample k unique indices

    directory = prefix.split('/')
    directory = '/'.join(directory[:-1])
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in idx:
        cap = captions[i].cpu().data.numpy()
        cap = ' '.join([vocab.idx2word[w] for w in cap])
        # print("%s" % cap)
        with open('%scap%d.txt' % (prefix, i), 'w') as fd:
            fd.write("%s\n" % cap)

        if use_mnist:
            img = images[i].cpu().data.numpy().reshape((28, 28))
            out = outputs[i].cpu().data.numpy().reshape((28, 28))
            plt.imsave('%simg%d.png' % (prefix, i), img, cmap='Greys')
            plt.imsave('%sout%d.png' % (prefix, i), out, cmap='Greys')
        else:
            # TODO: fix below! how to print proper RGB images..?
            img = np.uint8(
                images[i].cpu().data.numpy().reshape((256, 256, 3))*255
            )
            out = np.uint8(
                outputs[i].cpu().data.numpy().reshape((256, 256, 3))*255
            )
            plt.imsave('%simg%d.png' % (prefix, i), img)
            plt.imsave('%sout%d.png' % (prefix, i), out)

        # plt.imshow(img)
        # plt.imshow(out)


def run(args):
    """
    Main function!
    """
    vocab, train_loader, test_loader, embeddings = load_data(args)

    print("\nCreating encoder...")
    encoder = EncoderRNN(args.encoder_gate, embeddings, args.encoding_size, args.gaussian_dim, args.num_gaussians,
                         args.encoder_layers, args.dropout_rate, fixed_embeddings=args.fix_embeddings,
                         bidirectional=args.bidirectional, indep_gaussians=args.indep_gaussians)
    print(encoder)
    print("\nCreating decoder...")
    decoder = DecoderCNN(input_size=args.num_gaussians*args.gaussian_dim,
                         hidden_sizes=args.decoder_layers,
                         output_size=train_loader.dataset[0][0].view(-1).size(0))
    print(decoder)
    model_id = time.time()  # used to save the models

    # save parameters
    with open('%s_%s_args.pkl' % (args.save_prefix, model_id), 'wb') as f:
        pickle.dump(args, f)

    if torch.cuda.is_available():
        print("\ncuda available!")
        print("Moving variables to cuda %d..." % args.gpu)
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

    recon = torch.nn.MSELoss()

    start_time = time.time()
    best_valid = 100000.
    patience = args.patience

    print("\nTraining model...")
    for epoch in range(args.epochs):
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        nb_train_batches = 0.
        for i, (images, captions, lengths) in enumerate(train_loader):
            optimizer.zero_grad()

            images = to_var(images)
            captions = to_var(captions)

            sampled, mus, var = encoder(captions, lengths)
            outputs = decoder(sampled)

            kl_loss = kl(mus, var)
            recon_loss = recon(outputs, images)
            if args.verbose:
                print("epoch %.2d - step %.3d - kl loss %.6f - recon loss %.6f" % (
                    epoch+1, i+1, round(kl_loss.data[0], 6), round(recon_loss.data[0], 6)
                ))

            loss = kl_loss + 2*recon_loss  # put more emphasis on the reconstruction loss?
            loss.backward()
            optimizer.step()
            epoch_recon_loss += recon_loss.data[0]
            epoch_kl_loss += kl_loss.data[0]
            nb_train_batches += 1

            # in debug mode, break the training loop after 10 batches
            if args.debug and nb_train_batches == 11:
                break

        epoch_recon_loss /= nb_train_batches
        epoch_kl_loss /= nb_train_batches
        print("Epoch: %d - kl loss: %g - recon loss: %g" % (
            (epoch+1), round(epoch_kl_loss, 6), round(epoch_recon_loss, 6)
        ))

        # show some generated images after each epoch
        show_images(captions, images, outputs, args.use_mnist, vocab, k=1,
                prefix='%s_%s_samples/epoch%.2d' % (args.save_prefix, model_id, epoch+1)
        )

        print("computing validation loss...")
        valid_recon_loss = 0.0
        nb_valid_batches = 0.
        for i, (images, captions, lengths) in enumerate(test_loader):

            images = to_var(images)
            captions = to_var(captions)

            sampled, mus, var = encoder(captions, lengths)
            outputs = decoder(sampled)

            valid_recon_loss += recon(outputs, images).data[0]
            nb_valid_batches += 1

        valid_recon_loss /= nb_valid_batches

        print("valid loss: %g - best loss: %g" % (valid_recon_loss, best_valid))
        if valid_recon_loss < best_valid:
            best_valid = valid_recon_loss
            torch.save(encoder.state_dict(), "%s_%s_enc.pt" % (args.save_prefix, model_id))
            torch.save(decoder.state_dict(), "%s_%s_dec.pt" % (args.save_prefix, model_id))
            patience = args.patience  # reset patience
            print("Saved new model.")
        else:
            patience -= 1  # decrease patience
            print("No improvement. patience: %d" % patience)

        if patience <= 0:
            break

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
    parser.add_argument('--verbose', '-v', action='store_true', help="print accuracies at each step")
    parser.add_argument('--debug', '-d', action='store_true', help="stop training after 10 batches")
    parser.add_argument('--gpu', '-g', type=int, default=0, help="GPU id to run the model on if GPU is available")
    # training params
    parser.add_argument('--patience',      '-p',  type=int, default=10, help="number of epoch to continue training once validation loss decreases")
    parser.add_argument('--epochs',        '-e' , type=int, default=1, help="num epochs")
    parser.add_argument('--batch_size',    '-bs', type=int, default=32, help="mini batch size")
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--optimizer',     '-opt',choices=['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta'], default='adam', help="optimizer to use")
    # network architecture
    parser.add_argument('--num_gaussians', '-ng', type=int, default=2, help="num gaussians")
    parser.add_argument('--gaussian_dim',  '-gd', type=int, default=16, help="dimension of each gaussian variable")
    parser.add_argument('--activation',    '-ac', choices=['sigmoid', 'relu', 'swish'], type=str, default='relu', help="activation function")
    parser.add_argument('--dropout_rate',  '-dr', type=float, default=0.0, help="probability of dropout layer")
    parser.add_argument('--indep_gaussians', type=str2bool, default='True', help="Sample from independent gaussian distributions. Otherwise will sample from mixture of gaussians.")
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
    parser.add_argument('--vocab_loc', '-vl', default='./data/vocab.pkl', help="location of vocabulary")
    parser.add_argument('--use_mnist', type=str2bool, default='False', help="Use the MNIST dataset instead of the provided one")
    parser.add_argument('--images_loc', '-il', default='/coco/images/resized2014', help="location of resized images")
    parser.add_argument('--captions_loc', '-cl', default='/coco/annotations/captions_train2014.json', help="location of captions")
    parser.add_argument('--embeddings_loc', '-el', default=None, help="location of pretrained word embeddings")
    ## load & save model
    parser.add_argument('--save_prefix', default='models/VED', help="prefix to save model and arguments")
    parser.add_argument('--load_prefix', default=None, help="prefix to load model and arguments")
    args = parser.parse_args()
    print('args: %s\n' % args)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % args.gpu

    run(args)


