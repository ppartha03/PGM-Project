import time
import click
import pickle
import torch
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


@click.command()
@click.option('--epochs', '-e', default=1, help="num epochs")
@click.option('--learning_rate', '-lr', default=0.001, help="learning rate")
@click.option('--num_gaussians', '-g', default=32, help="num gaussians")
@click.option('--images_loc', '-i', default='/coco/images/resized2014', help="location of resized images")
@click.option('--captions_loc', '-c', default='/coco/annotations/captions_train2014.json', help="location of captions")
@click.option('--vocab_loc', '-v', default='./data/vocab.pkl', help="location of vocabulary")
def run(epochs, num_gaussians, learning_rate, images_loc, captions_loc, vocab_loc):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    with open(vocab_loc, 'rb') as f:
        vocab = pickle.load(f)

    train_loader = get_loader(root=images_loc,
                              json=captions_loc,
                              vocab=vocab, batch_size=32, num_workers=2, shuffle=True, transform=transform)

    encoder = EncoderRNN(vocab_size=len(vocab), hidden_size=32, embed_size=16, num_gaussians=num_gaussians)
    decoder = DecoderCNN(input_size=num_gaussians,
                         output_size=train_loader.dataset[0][0].view(-1).size(0))

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params=params, lr=learning_rate)

    #  TODO add KL with covarariance matrix
    def kl(mu, log_var): return (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())) / mu.view(-1).size(0)

    recon = torch.nn.MSELoss()

    start_time = time.time()
    print("training model")
    for epoch in range(epochs):
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

            loss = kl_loss + recon_loss
            loss.backward()
            optimizer.step()
            epoch_recon_loss += recon_loss.data[0]
            epoch_kl_loss += kl_loss.data[0]
            nb_train_batches += 1

        epoch_recon_loss /= nb_train_batches
        epoch_kl_loss /= nb_train_batches
        print("Epoch: ", epoch, "recon loss: ", round(epoch_recon_loss, 4),
              "kl loss: ", round(epoch_kl_loss))

    print('Finished Training, time elapsed: ', round(time.time() - start_time, 2), ' seconds')


if __name__ == '__main__':
    run()


