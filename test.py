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

    for i in idx:
        cap = captions[i].cpu().data.numpy()
        cap = ' '.join([vocab.idx2word[w] for w in cap])
        # print("%s" % cap)
        with open('%s_cap%d.txt' % (prefix, i), 'w') as fd:
            fd.write("%s\n" % cap)

        if use_mnist:
            img = images[i].cpu().data.numpy().reshape((28, 28))
            out = outputs[i].cpu().data.numpy().reshape((28, 28))
            plt.imsave('%s_img%d.png' % (prefix, i), img, cmap='Greys')
            plt.imsave('%s_out%d.png' % (prefix, i), out, cmap='Greys')
        else:
            # TODO: fix below! how to print proper RGB images..?
            img = np.uint8(
                images[i].cpu().data.numpy().reshape((256, 256, 3))*255
            )
            out = np.uint8(
                outputs[i].cpu().data.numpy().reshape((256, 256, 3))*255
            )
            plt.imsave('%s_img%d.png' % (prefix, i), img)
            plt.imsave('%s_out%d.png' % (prefix, i), out)

        # plt.imshow(img)
        # plt.imshow(out)


def kl(mu, log_var):
    return (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())) / mu.view(-1).size(0)


def run(args):
    """
    Main function!
    """
    with open('%s_args.pkl' % args.load_prefix, 'rb') as f:
        old_args = pickle.load(f)

    vocab, train_loader, test_loader, embeddings = load_data(old_args)

    print("\nCreating encoder...")
    encoder = EncoderRNN(old_args.encoder_gate, embeddings, old_args.encoding_size, old_args.gaussian_dim,
                         old_args.num_gaussians, old_args.encoder_layers, old_args.dropout_rate,
                         fixed_embeddings=old_args.fix_embeddings, bidirectional=old_args.bidirectional,
                         indep_gaussians=old_args.indep_gaussians)
    encoder.load_state_dict(torch.load("%s_enc.pt" % args.load_prefix))
    print(encoder)
    print("\nCreating decoder...")
    decoder = DecoderCNN(input_size=old_args.num_gaussians*old_args.gaussian_dim,
                         hidden_sizes=old_args.decoder_layers,
                         output_size=train_loader.dataset[0][0].view(-1).size(0))
    decoder.load_state_dict(torch.load("%s_dec.pt" % args.load_prefix))
    print(decoder)

    if torch.cuda.is_available():
        print("\ncuda available!")
        print("Moving variables to cuda %d..." % torch.cuda.current_device())
        encoder.cuda()
        decoder.cuda()

    recon = torch.nn.MSELoss()

    test_recon_loss = 0.
    test_kl_loss = 0.
    n_batches = 0.
    start_time = time.time()
    for i, (images, captions, lengths) in enumerate(test_loader):

        images = to_var(images)
        captions = to_var(captions)

        sampled, mus, var = encoder(captions, lengths)
        outputs = decoder(sampled)

        kl_loss = kl(mus, var)
        recon_loss = recon(outputs, images)

        test_recon_loss += recon_loss.data[0]
        test_kl_loss += kl_loss.data[0]
        n_batches += 1

    test_recon_loss /= n_batches
    test_kl_loss /= n_batches
    print("test kl loss: %g - test recon loss: %g" % (
        round(test_kl_loss, 6), round(test_recon_loss, 6)
    ))

    # show some generated images after each epoch
    show_images(captions, images, outputs, old_args.use_mnist, vocab, k=10,
            prefix='%s_samples/test' % args.load_prefix
    )

    print('Finished Testing, time elapsed: ', round(time.time() - start_time, 2), ' seconds')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('load_prefix', help="prefix to load model and arguments")
    parser.add_argument('--verbose', '-v', action='store_true', help="print accuracies at each step")
    parser.add_argument('--debug', '-d', action='store_true', help="stop training after 10 batches")
    parser.add_argument('--gpu', '-g', type=int, default=0, help="GPU id to run the model on if GPU is available")
    args = parser.parse_args()
    print('args: %s\n' % args)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % args.gpu

    run(args)


