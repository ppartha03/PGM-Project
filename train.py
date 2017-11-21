import time
import pickle
import argparse
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


def run(args):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    with open(args.vocab_loc, 'rb') as f:
        vocab = pickle.load(f)

    train_loader = get_loader(root=args.images_loc,
                              json=args.captions_loc,
                              vocab=vocab, batch_size=1024, num_workers=2, shuffle=True, transform=transform)

    print("\nCreating encoder...")
    encoder = EncoderRNN(vocab_size=len(vocab), embed_size=16, hidden_size=32, gaussian_dim=args.gaussian_dim, num_gaussians=args.num_gaussians)
    print("Creating decoder...")
    decoder = DecoderCNN(input_size=args.num_gaussians*args.gaussian_dim,
                         output_size=train_loader.dataset[0][0].view(-1).size(0))

    if torch.cuda.is_available():
        print("\ncuda available!")
        print("Moving variables to cuda %d..." % torch.cuda.current_device())
        encoder.cuda()
        decoder.cuda()

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params=params, lr=args.learning_rate)

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

            if args.debug and nb_train_batches == 10:
                break

        epoch_recon_loss /= nb_train_batches
        epoch_kl_loss /= nb_train_batches
        print("Epoch: ", epoch, "recon loss: ", round(epoch_recon_loss, 4),
              "kl loss: ", round(epoch_kl_loss, 4))

    print('Finished Training, time elapsed: ', round(time.time() - start_time, 2), ' seconds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', '-e', type=int, default=1, help="num epochs")
    parser.add_argument('--debug', '-d', action='store_true', help="stop training after 10 batches")
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--num_gaussians', '-ng', type=int, default=2, help="num gaussians")
    parser.add_argument('--gaussian_dim', '-gd', type=int, default=16, help="dimension of each gaussian variable")
    parser.add_argument('--images_loc', '-i', default='/coco/images/resized2014', help="location of resized images")
    parser.add_argument('--captions_loc', '-c', default='/coco/annotations/captions_train2014.json', help="location of captions")
    parser.add_argument('--vocab_loc', '-v', default='./data/vocab.pkl', help="location of vocabulary")
    args = parser.parse_args()
    print('args: %s\n' % args)

    run(args)


