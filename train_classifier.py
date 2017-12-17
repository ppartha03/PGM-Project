import argparse
import pickle
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

from classifier import Net


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help="gpu id to use")
parser.add_argument('--batch_size', type=int, default=128, help="batch size to use")
parser.add_argument('--learning_rate', type=float, default=0.001, help="Adam optimizer learning rate")
parser.add_argument('--patience', type=int, default=10, help="patience")
args = parser.parse_args()
print("args: %s\n" % args)
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % args.gpu


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size,
    shuffle=False
)


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


model = Net()
print(model)
model_id = time.time()

# save parameters
with open('./models/%s_args.pkl' % model_id, 'wb') as f:
    pickle.dump(args, f)

# move model to gpu
if torch.cuda.is_available():
    print("\ncuda available! moving model to gpu %d" % args.gpu)
    model.cuda()

# Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# Negative log likelihood
loss = torch.nn.CrossEntropyLoss()

start_time = time.time()
best_valid = 0.0
patience = args.patience

print("\nTraining model...")
for epoch in range(1000):  # max epoch 1,000
    epoch_loss = 0.0
    epoch_correct = 0.0
    nb_train_batches = 0.0
    for b_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        images = to_var(images)
        labels = to_var(labels)

        outputs = model(images)
        pred_loss = loss(outputs, labels)
        pred_loss.backward()
        optimizer.step()

        epoch_loss += pred_loss.data[0]
        predictions = outputs.data.max(1, keepdim=True)[1]  # idx of max log prob
        epoch_correct += predictions.eq(labels.data.view_as(predictions)).cpu().sum()
        nb_train_batches += 1

    epoch_loss /= nb_train_batches
    epoch_correct /= len(train_loader.dataset)
    print("Epoch: %d - loss: %g - accuracy: %g" % (epoch+1, epoch_loss, epoch_correct))

    print("Computing validation loss...")
    valid_loss = 0.0
    valid_correct = 0.0
    nb_valid_batches = 0.0
    for b_idx, (images, labels) in enumerate(test_loader):
        images = to_var(images)
        labels = to_var(labels)

        outputs = model(images)

        valid_loss += loss(outputs, labels).data[0]
        predictions = outputs.data.max(1, keepdim=True)[1]  # idx of max log prob
        valid_correct += predictions.eq(labels.data.view_as(predictions)).cpu().sum()
        nb_valid_batches += 1

    valid_loss /= nb_valid_batches
    valid_correct /= len(test_loader.dataset)
    print("valid loss: %g - valid accuracy: %g - best accuracy: %g" % (valid_loss, valid_correct, best_valid))
    if valid_correct > best_valid:
        best_valid = valid_correct
        torch.save(model.state_dict(), "./models/%s_classifier.pt" % model_id)
        patience = args.patience
        print("Saved new model.")
    else:
        patience -= 1
        print("No improvement. patience: %d" % patience)

    if patience <= 0:
        break

print("Finished training, time elapsed:", round(time.time() - start_time, 2), "seconds")



