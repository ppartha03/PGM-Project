import nltk
import pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab():
    """Build MNIST vocabulary wrapper."""

    # Creates a vocab wrapper and add some special tokens
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # digit words
    digits = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    for w in digits:
        vocab.add_word(w)

    # sentence words
    words = ['this', 'is', 'the', 'a', 'that', "'s", 'number', 'black',
            'on', 'white', 'background', 'with']
    '''
    possible sentences:
    > that 's ...
    > this is a ...
    > this number is ...
    > this is the number ...
    '''
    for w in words:
        vocab.add_word(w)

    # some logical connectors (maybe for future experiments)
    logicals = ['min', 'max', 'or', 'plus', 'minus']
    '''
    possible combinations:
    > min one four       --> should draw a 1
    > max one four       --> should draw a 4
    > max one five three --> should draw a five
    > one or two         --> should draw either a 1 or a 2
    '''
    for w in logicals:
        vocab.add_word(w)

    return vocab


def main(args):
    vocab = build_vocab()
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: %d" % len(vocab))
    print("Saved the vocabulary wrapper to '%s'" % vocab_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_path', type=str, default='./data/mnist_vocab.pkl',
                        help='path for saving vocabulary wrapper')
    args = parser.parse_args()
    main(args)

