import torch
from torchvision import transforms, datasets
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths


def get_loader(root, json, vocab, train, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    if train:
        root = root.replace('val', 'train')
        json = json.replace('val', 'train')
    else:
        root = root.replace('train', 'val')
        json = json.replace('train', 'val')
    coco = CocoDataset(root=root,
                       json=json,
                       vocab=vocab,
                       transform=transform)

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for every iteration.
    # images: tensor of shape (batch_size, 3, 224, 224).
    # captions: tensor of shape (batch_size, padded_length).
    # lengths: list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader



MNIST_VOCAB = None
MNIST_DIST = None

def mnist_collate_fn(data):
    """
    Used for MNIST
    transforms data[1] to be a list of captions instead of a list of labels
    """
    assert MNIST_VOCAB is not None
    assert MNIST_DIST is not None

    sents_lvl1 = ['<label>']
    sents_lvl2 = ["that 's <label>",
                  "that 's a <label>"
                  "this is a <label>",
                  "this number is <label>",
                  "this number is a <label>",
                  "this is the number <label>",
                  "black <label> on white"
                  "black <label> on white background",
                  "that 's a <label> on a white background",
                  "that is a black <label> with a white background",
                  "this is a black <label> on a white background"]

    label2word = ['zero', 'one', 'two', 'three', 'four', 'five', 'six',
            'seven', 'eight', 'nine']

    images, labels = zip(*data)
    for idx, l in enumerate(labels):
        if MNIST_DIST < 1:
            # sent = np.random.choice(sents_lvl1)
            sent = np.random.choice(sents_lvl1 + sents_lvl2)
        else:
            # create a logical sentence, ex: ``min six five nine``
            if int(l) + MNIST_DIST > len(label2word)-1:
                sent = 'max '
            elif int(l) - MNIST_DIST < 0:
                sent = 'min '
            else:
                sent = np.random.choice(['min ', 'max '])  # start with min/max

            if sent == 'max ':  # sentence starts with `max`
                # candidate distractions must be smaller than the label
                candidates = label2word[:int(l)]
            else:  # sentence starts with `min`
                # candidate distractions must be higher than the label
                candidates = label2word[int(l)+1:]
            assert len(candidates) >= MNIST_DIST  # make sure we have enough to sample

            content = ['<label>']  # list of numbers
            # extend list of numbers with distractions
            content.extend(np.random.choice(candidates, MNIST_DIST, replace=False))
            np.random.shuffle(content)  # shuffle the list of numbers
            sent += ' '.join(content)

        # replace placeholder by the actual label
        sent = sent.replace('<label>', label2word[int(l)])

        # create caption
        tokens = nltk.tokenize.word_tokenize(str(sent).lower())
        caption = []
        caption.append(MNIST_VOCAB('<start>'))
        caption.extend([MNIST_VOCAB(token) for token in tokens])
        caption.append(MNIST_VOCAB('<end>'))
        target = torch.Tensor(caption)

        # set the caption in the data
        data[idx] = (data[idx][0], target)

    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths

def get_mnist_loader(vocab, train, download, transform, batch_size, dist, shuffle, num_workers):
    """
    :param vocab: vocabulary of the captions
    :param train: data flag mode
    :param download: flag to download or not
    :param transform: image transformer
    :param batch_size: number of examples per batch
    :param dist: if > 1 then using logical captions with this amount of distraction
    :param shuffke: flag to shuffle the mini batches
    :param num_workers: number of threads for loading data
    """
    # MNIST data
    mnist = datasets.MNIST(
            './data',
            train=train,
            download=download,
            transform=transform)
    global MNIST_VOCAB
    MNIST_VOCAB = vocab
    global MNIST_DIST
    MNIST_DIST = dist

    data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=mnist_collate_fn)
    return data_loader

