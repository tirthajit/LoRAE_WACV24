import os
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import argparse
from tqdm import tqdm
import model
import utils

parser = argparse.ArgumentParser(description="Training Autoencoders")

parser.add_argument('-n', type=int, help='latent dimension', default=256)

parser.add_argument('-l', type=int, help='implicit layer', default=1)

parser.add_argument('--epochs', type=int, help='#epochs', default=100)

parser.add_argument('--dataset', type=str, default="celeba")

parser.add_argument('--optimizer', type=str, default="adam")

parser.add_argument('--nuc-lambda', type=int, help='Nuc lambda index [0.001, 0.01, 0.1, 1, 5]', default=9)

parser.add_argument('--vanilla', action='store_true', help='VANILLA')

# Let following arguments be in default
parser.add_argument('--batch-size', type=int, default=100) #32 earlier
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--beta', type=float, default=0.1)
parser.add_argument('--checkpoint', type=str, default="./checkpoint/")
parser.add_argument('--data-path', type=str, default="./data/")


def main(args):

    # use gpu ##########################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset ##########################################
    if args.dataset == "mnist":
        args.data_path = args.data_path + "mnist/"
        if not os.path.exists(args.data_path):
            os.makedirs(args.data_path)

        train_set = datasets.MNIST(args.data_path, train=True,
                                   download=True,
                                   transform=transforms.Compose([
                                     transforms.Resize([32, 32]),
                                     transforms.ToTensor()]))
        valid_set = datasets.MNIST(args.data_path, train=False,
                                   download=True,
                                   transform=transforms.Compose([
                                     transforms.Resize([32, 32]),
                                     transforms.ToTensor()]))
        
    elif args.dataset == "fmnist":
        args.data_path = args.data_path + "fmnist/"
        if not os.path.exists(args.data_path):
            os.makedirs(args.data_path)

        train_set = datasets.FashionMNIST(args.data_path, train=True,
                                   download=True,
                                   transform=transforms.Compose([
                                     transforms.Resize([32, 32]),
                                     transforms.ToTensor()]))
        valid_set = datasets.FashionMNIST(args.data_path, train=False,
                                   download=True,
                                   transform=transforms.Compose([
                                     transforms.Resize([32, 32]),
                                     transforms.ToTensor()]))
    elif args.dataset == "intel":
        train_set = ImageFolder(
            args.data_path + 'intel/train/',
            transform=transforms.Compose([transforms.Resize([64, 64]),
                                          transforms.ToTensor()]))
        valid_set = ImageFolder(
            args.data_path + 'intel/test/',
            transform=transforms.Compose([transforms.Resize([64, 64]),
                                          transforms.ToTensor()]))
    
    elif args.dataset == "cifar10":
        train_set = datasets.CIFAR10(
            args.data_path + '/cifar10/',
            train=True,
            transform=transforms.Compose([transforms.Resize([64, 64]),
                                          transforms.ToTensor()]))
        valid_set = datasets.CIFAR10(
            args.data_path + '/cifar10/',
            train=False,
            transform=transforms.Compose([transforms.Resize([64, 64]),
                                          transforms.ToTensor()]))

    elif args.dataset == "celeba":
        train_set = utils.ImageFolder(
            args.data_path + 'celeba/train/',
            transform=transforms.Compose([transforms.CenterCrop(148), #erlier 148
                                          transforms.Resize([64, 64]),
                                          transforms.ToTensor()]))
        valid_set = utils.ImageFolder(
            args.data_path + 'celeba/val/',
            transform=transforms.Compose([transforms.CenterCrop(148),
                                          transforms.Resize([64, 64]),
                                          transforms.ToTensor()]))
        
    elif args.dataset == "shape":
        train_set = utils.ShapeDataset(
            data_size=50000)
        valid_set = utils.ShapeDataset(
            data_size=10000)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        num_workers=32,
        batch_size=args.batch_size
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        num_workers=32,
        batch_size=args.batch_size
    )

    # init networks ##########################################
    net = model.AE(args)
    net.to(device)
    
    n_lambda= [0.001, 0.01, 0.1, 1, 5, 0.005, 0.003, 0.0001, 0.0005, 0.00001]
    
    # Define the nuclear norm regularization strength
    nuclear_norm_strength = n_lambda[args.nuc_lambda]
    
    # optimizer ##########################################
    if args.optimizer=='adam':
        optimizer = optim.Adam(net.parameters(), args.lr)
    elif args.optimizer=='sgd':
        optimizer= optim.SGD(net.parameters(), args.lr)
    elif args.optimizer=='adagrad':
        optimizer= optim.Adagrad(net.parameters(), args.lr)
        
    # train ################################################
    save_path = args.checkpoint + "/" + args.dataset + "/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for e in range(args.epochs):
        
        recon_loss = 0

        for yi, _, in tqdm(train_loader):
            net.train()

            optimizer.zero_grad()

            yi = yi.to(device)
            loss = net(yi)

            mat= net.mlp.hidden[0].weight

            # Compute the nuclear norm regularization term
            nuclear_norm_regularization = torch.linalg.norm(mat, ord='nuc')

            loss += nuclear_norm_strength * nuclear_norm_regularization
            
            recon_loss += loss.item()
            loss.backward()
            optimizer.step()

        recon_loss /= len(train_loader)
        print("epoch " + str(e) + '\ttraining loss = ' + str(recon_loss))

        # save model #########################################

        if args.vanilla == True:
            torch.save(net.state_dict(), save_path+ 'l' + str(args.l) + '_n' + str(args.n) + '_' + str(args.optimizer) + 'vanilla')
        else:
            torch.save(net.state_dict(), save_path+ 'l' + str(args.l) + '_n' + str(args.n) + '_' + str(args.optimizer) + '_nuc_lambda_index' + str(args.nuc_lambda))


        valid_loss = 0

        for yi, _ in tqdm(valid_loader):
            net.eval()

            yi = yi.to(device)
            eval_loss = net(yi)
            valid_loss += eval_loss.item()

        valid_loss /= len(valid_loader)

        print("epoch " + str(e) + '\tvalid loss = ' + str(valid_loss))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
