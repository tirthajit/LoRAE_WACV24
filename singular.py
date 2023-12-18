import os
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
import model
import utils

parser = argparse.ArgumentParser(description="interpolation")
parser.add_argument('-n', type=int, help='latent dimension', default=128)
parser.add_argument('-l', type=int, help='layers', default=1)

parser.add_argument('--dataset', type=str, default="mnist")

parser.add_argument('--optimizer', type=str, default="adam")

parser.add_argument('--nuc-lambda', type=int, help='Nuc lambda index [0.001, 0.01, 0.1, 1, 5]', default=0)

parser.add_argument('--irmae', action='store_true', help='IRMAE')
parser.add_argument('--ours', action='store_true', help='OURS')
parser.add_argument('--checkpoint', type=str, default="./checkpoint/")
parser.add_argument('--data-path', type=str, default="../data/")
parser.add_argument('--save-path', type=str, default="./results/")


def main(args):

    # dataset ##########################################
    if args.dataset == "mnist":
        args.data_path = args.data_path + "mnist/"
        if not os.path.exists(args.data_path):
            os.makedirs(args.data_path)

        test_set = datasets.MNIST(args.data_path, train=False,
                                  transform=transforms.Compose([
                                     transforms.Resize([32, 32]),
                                     transforms.ToTensor()]))

        test_loader = torch.utils.data.DataLoader(
            test_set,
            num_workers=32,
            batch_size=100
        )


    elif args.dataset == "fmnist":
        args.data_path = args.data_path + "fmnist/"
        if not os.path.exists(args.data_path):
            os.makedirs(args.data_path)

        test_set = datasets.FashionMNIST(args.data_path, train=False,
                                  transform=transforms.Compose([
                                     transforms.Resize([32, 32]),
                                     transforms.ToTensor()]))

        test_loader = torch.utils.data.DataLoader(
            test_set,
            num_workers=32,
            batch_size=100
        )
    elif args.dataset=='svhn':
        args.data_path = args.data_path + "svhn/"
        if not os.path.exists(args.data_path):
            os.makedirs(args.data_path)
        test_set= datasets.SVHN(args.data_path, split= 'test', 
                              transform=transforms.Compose([transforms.Resize([64, 64]), 
                                                            transforms.ToTensor()])
                            )
        test_loader = torch.utils.data.DataLoader(
            test_set,
            num_workers=32,
            batch_size=100
        )
    
    elif args.dataset == "intel":
        test_set = datasets.ImageFolder(
            args.data_path + 'intel/test/',
            transform=transforms.Compose([transforms.Resize([64, 64]),
                                          transforms.ToTensor()]))
        
        test_loader = torch.utils.data.DataLoader(
            test_set,
            num_workers=32,
            batch_size=100
        )

    elif args.dataset == "cifar10":
        test_set = datasets.CIFAR10(
            args.data_path + '/cifar10/',
            train=False,
            transform=transforms.Compose([transforms.Resize([64, 64]),
                                          transforms.ToTensor()]))
        
        test_loader = torch.utils.data.DataLoader(
            test_set,
            num_workers=32,
            batch_size=100
        )
    
    elif args.dataset == "celeba":
        test_set = utils.ImageFolder(
            args.data_path + 'celeba/test/',
            transform=transforms.Compose([transforms.CenterCrop(148),
                                          transforms.Resize([64, 64]),
                                          transforms.ToTensor()]))
        test_loader = torch.utils.data.DataLoader(
            test_set,
            num_workers=32,
            batch_size=100
        )

    # load model ##########################################
    net = model.AE(args)

    if args.l > 0 and args.irmae == True:
        net.load_state_dict(torch.load(args.checkpoint + "/" + args.dataset + "/"
                        + 'irmae_l'+str(args.l)+'_'+str(args.n),
                        map_location=torch.device('cpu')))
    
    elif args.l > 0 and args.ours == True:
        net.load_state_dict(torch.load(args.checkpoint + "/" + args.dataset + "/"
                        + 'l' + str(args.l) + '_n' + str(args.n) + '_' + str(args.optimizer) + '_nuc_lambda_index' + str(args.nuc_lambda),
                        map_location=torch.device('cpu')))      
    
    else:
        net.load_state_dict(torch.load(args.checkpoint + "/" + args.dataset + "/"
                        + 'vanilla_'+str(args.n)))
           
    net.eval()

    z = []
    for yi, _ in test_loader:
        z_hat = net.encode(yi)
        z.append(z_hat)

    z = torch.cat(z, dim=0).data.numpy()
    c = np.cov(z, rowvar=False)
    u, d, v = np.linalg.svd(c)
    

    d = d / d[0]
    
    plt.figure(figsize=(6, 4))
    plt.plot(range(args.n), d)
    plt.autoscale(enable=True, axis='y', tight=True)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.ylim(0, 0.4)
    plt.xlim(0, args.n)
    plt.xlabel("Singular Value Rank")
    plt.ylabel("Singular Values")
    plt.title("Singular Values of Covariance Matrix")

    if args.dataset == "shape":
        plt.axvline(x=7, color='k', linestyle='dashed', linewidth=1)

    path = args.save_path + "/" + args.dataset + "/singular/"
    if not os.path.exists(path):
        os.makedirs(path)
    
    if args.l > 0 :
        plt.savefig(args.save_path + '/' + args.dataset + '/singular/'
                        + 'l' + str(args.l) + '_n' + str(args.n) + '_' + str(args.optimizer) + '_nuc_lambda_index' + str(args.nuc_lambda) + '.png', bbox_inches='tight')
    else:
        plt.savefig(args.save_path + '/' + args.dataset + '/singular/'
                        + 'l' + str(args.l) + '_n' + str(args.n) + '_' + str(args.optimizer) + 'vanilla' + '.png', bbox_inches='tight')
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
