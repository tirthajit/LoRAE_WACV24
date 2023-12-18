import os
import numpy as np
import argparse
import torch
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn import mixture
import matplotlib.pyplot as plt
import model
import model_vae
from PIL import Image
import utils

parser = argparse.ArgumentParser(description="Generative and Downstream Tasks")
parser.add_argument('--dataset', type=str, default="mnist")
parser.add_argument('--task', type=str, default="gmm")
parser.add_argument('--fid', action='store_true', help='Calculate FID score')
parser.add_argument('-n', type=int, help='latent dimension', default=128)
parser.add_argument('-l', type=int, help='layers', default=1)

parser.add_argument('--vae', action='store_true', help='VAE')
 
parser.add_argument('--vanilla', action='store_true', help='VANILLA')

parser.add_argument('--irmae', action='store_true', help='IRMAE')


parser.add_argument('--optimizer', type=str, default="adam")

parser.add_argument('--nuc-lambda', type=int, help='Nuc lambda index [0.001, 0.01, 0.1, 1, 5]', default=0)

parser.add_argument('-d', type=int, help='PCA dimension', default=4)
parser.add_argument('-X', type=int, default=10)
parser.add_argument('-Y', type=int, default=10)
parser.add_argument('-N', type=int, default=1000)
parser.add_argument('--test-size', type=int, default=1000)

parser.add_argument('--sample', type=int, default=20000)
parser.add_argument('--do_sample', action='store_true', help='DOES SAMPLING')

parser.add_argument('--checkpoint', type=str, default="./checkpoint/")
parser.add_argument('--data-path', type=str, default="../data/")
parser.add_argument('--save-path', type=str, default="./results/")


def main(args):

    # dataset ##########################################
    if args.dataset == "mnist":
        args.data_path = args.data_path + "mnist/"
        if not os.path.exists(args.data_path):
            os.makedirs(args.data_path)

        test_set = datasets.MNIST(args.data_path, train=True,
                                  transform=transforms.Compose([
                                     transforms.Resize([32, 32]),
                                     transforms.ToTensor()]))

        test_loader = torch.utils.data.DataLoader(
            test_set,
            num_workers=32,
            batch_size=args.test_size
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
            batch_size=args.test_size
        )
    elif args.dataset == "intel":
        test_set = datasets.ImageFolder(
            args.data_path + 'intel/test/',
            transform=transforms.Compose([transforms.Resize([64, 64]),
                                          transforms.ToTensor()]))
        
        test_loader = torch.utils.data.DataLoader(
            test_set,
            num_workers=32,
            batch_size=args.test_size
        )
    elif args.dataset == "cifar10":
        test_set = datasets.CIFAR10(
            args.data_path + 'cifar10/',
            train=False,
            transform=transforms.Compose([transforms.Resize([64, 64]),
                                          transforms.ToTensor()]))
        
        test_loader = torch.utils.data.DataLoader(
            test_set,
            num_workers=32,
            batch_size=args.test_size
        )

    elif args.dataset == "celeba":
        test_set = utils.ImageFolder(
            args.data_path + 'celeba/test/',
            transform=transforms.Compose([transforms.CenterCrop(148),
                                          transforms.Resize([64, 64]),
                                          transforms.ToTensor()]))
    elif args.dataset == "shape":
        test_set = utils.ShapeDataset(
            data_size=10000)
        
    if args.do_sample:
        test_loader = torch.utils.data.DataLoader(
            test_set,
            num_workers=32,
            batch_size=args.test_size,
            sampler=torch.utils.data.SubsetRandomSampler(
            range(args.sample))
        )
    else:
        test_loader = torch.utils.data.DataLoader(
            test_set,
            num_workers=32,
            batch_size=args.test_size
            )
    
    # load model ##########################################
    if args.vae:
        net= model_vae.AE(args)
    else:
        net = model.AE(args)

    
    path = args.save_path + args.dataset + "/" + args.task
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    if args.vae:
        net.load_state_dict(torch.load(args.checkpoint + "/" + args.dataset
                        + '/vae',
                        map_location=torch.device('cpu')))
        path += '/vae' + '_d' +str(args.d)

    elif args.vanilla:
        net.load_state_dict(torch.load(args.checkpoint + "/" + args.dataset
                        + '/vanilla_'+str(args.n),
                        map_location=torch.device('cpu')))
        path += '/vanilla' + '_d' +str(args.d)
    
    elif args.irmae:
        net.load_state_dict(torch.load(args.checkpoint + "/" + args.dataset + "/"
                        + 'irmae_l'+ str(args.l) + '_' + str(args.n),
                        map_location=torch.device('cpu')))
        path += '/' + 'irmae_l' + str(args.l) + '_d' +str(args.d) + '_' + str(args.n)

    else:
        net.load_state_dict(torch.load(args.checkpoint + "/" + args.dataset + "/"
                        + 'l' + str(args.l) + '_n' + str(args.n) + '_' + str(args.optimizer) + '_nuc_lambda_index' + str(args.nuc_lambda),
                        map_location=torch.device('cpu')))
        path += '/' + 'l' + str(args.l) + '_n' + str(args.n) + '_' + str(args.optimizer) + '_nuc_lambda_index' + str(args.nuc_lambda) + '_d' +str(args.d)
    
    
    net.eval()

    fig, axs = plt.subplots(args.X, args.Y, figsize=[args.Y, args.X])

    if args.task == "reconstruction":
        yi, _ = next(iter(test_loader))
        zi = net.encode(yi)

        y_hat = net.decode(zi[:args.X * args.Y]).data.numpy()

    elif args.task == "interpolation":
        yi, _ = next(iter(test_loader))

        zi = net.encode(yi)

        zs = []
        for i in range(args.X):
            z0 = zi[i*2]
            z1 = zi[i*2+1]

            for j in range(args.Y):
                zs.append((z0 - z1) * j / args.Y + z1)
        zs = torch.stack(zs, axis=0)
        y_hat = net.decode(zs).data.numpy()

    elif args.task == "mvg":
        z = []
        for yi, _ in test_loader:
            zi = net.encode(yi)
            z.append(zi.detach().numpy())
        z = np.concatenate(z, axis=0)
        mu = np.average(z, axis=0)
        sigma = np.cov(z, rowvar=False)

        # generate corresponding sample z
        if args.fid:
            if args.dataset=='intel':
                zs = np.random.multivariate_normal(mu, sigma, 3000)
            elif args.dataset=='cifar10':
                zs = np.random.multivariate_normal(mu, sigma, 5000)
            elif args.dataset=='shape':
                zs = np.random.multivariate_normal(mu, sigma, 10000)
            elif args.dataset=='mnist' or args.dataset=='fmnist':
                zs = np.random.multivariate_normal(mu, sigma, 10000)
            elif args.dataset=='celeba':
                zs = np.random.multivariate_normal(mu, sigma, args.sample)
        else:
            zs = np.random.multivariate_normal(mu, sigma, args.X * args.Y)
        
        zs = torch.Tensor(zs)

        if args.fid==True:

            y_hat= []

            for i in range(int(zs.shape[0]/args.test_size)):
                
                k= i*args.test_size

                y_temp= net.decode(zs[k:k+args.test_size , :]).data.numpy()

                y_hat.append(y_temp)
            
            y_hat = np.concatenate(y_hat, axis=0)
            
        else:
            y_hat = net.decode(zs).data.numpy()
    
    elif args.task == "gmm":
        z = []
        for yi, _ in test_loader:
            zi = net.encode(yi)
            z.append(zi.detach().numpy())
        z = np.concatenate(z, axis=0)
        gmm = mixture.GaussianMixture(
            n_components=args.d, covariance_type='full')

        gmm.fit(z)

        if args.fid:

            if args.dataset=='intel':
                zs, _ = gmm.sample(3000)
            elif args.dataset=='cifar10':
                zs,_ = gmm.sample(5000)
            
            elif args.dataset=='shape':
                zs,_ = gmm.sample(10000)
            
            elif args.dataset=='mnist' or args.dataset=='fmnist':
                zs,_ = gmm.sample(10000)
            elif args.dataset=='celeba':
                zs,_ = gmm.sample(args.sample)

        else:
            zs, _ = gmm.sample(args.X * args.Y)

        zs = torch.Tensor(zs)

        if args.fid==True:

            y_hat= []

            for i in range(int(zs.shape[0]/args.test_size)):
                
                k= i*args.test_size

                y_temp= net.decode(zs[k:k+args.test_size , :]).data.numpy()

                y_hat.append(y_temp)
            
            y_hat = np.concatenate(y_hat, axis=0)

        else:
            y_hat = net.decode(zs).data.numpy()

    elif args.task == "pca":
        z = []
        for yi, _ in test_loader:
            zi = net.encode(yi)
            z.append(zi.detach().numpy())
        z = np.concatenate(z, axis=0)

        pca = PCA(n_components=args.d)
        pca.fit(z)
        x = pca.transform(z)
        mu = np.average(x, axis=0)
        sigma = np.cov(x, rowvar=False)

        sigma_0 = np.sqrt(sigma[0][0])
        sigma_1 = np.sqrt(sigma[1][1])
        center = mu.copy()
        center[0] -= sigma_0 * 2
        center[1] -= sigma_1 * 2

        zs = []
        for i in range(args.X):
            tmp = []
            x = center.copy()
            x[0] += sigma_0 * i / args.X * 4
            for j in range(args.Y):
                x[1] += sigma_1 / args.Y * 4
                zi = pca.inverse_transform(x)
                tmp.append(zi)
            tmp = np.stack(tmp, axis=0)
            zs.append(tmp)
        zs = np.concatenate(zs, axis=0)
        zs = torch.Tensor(zs)

        y_hat = net.decode(zs).data.numpy()

    if args.fid:

        test_data_path= './test_image_folder/'+ args.dataset
        
        if not os.path.exists(test_data_path):
            os.makedirs(test_data_path)
            real_hat=[]
            
            if args.dataset== 'cifar10':
                # Define the number of images per class you want to use
                num_images_per_class = 500
                # Create a new dataset with a subset of images per class
                subset = []
                class_count = [0] * 10  # To keep track of the number of images per class

                for i in range(len(test_set)):
                    image, label = test_set[i]
                    if class_count[label] < num_images_per_class:
                        subset.append((image, label))
                        class_count[label] += 1
                    if sum(class_count) == 10 * num_images_per_class:
                        break  
                test_loader= torch.utils.data.DataLoader(subset,
                                                        num_workers=32,
                                                        batch_size=args.test_size
                                                        )

            for re, label in test_loader:
                img= re.numpy()
                real_hat.append(img)
            
            real_hat= np.concatenate(real_hat, 0)
            
            
            # Iterate over the stacked images and save each one individually
            for i, image in enumerate(real_hat):
                
                if args.dataset=='mnist' or args.dataset=='fmnist':
                    
                    # Reshape the image from (1, 28, 28) to (28, 28)
                    image = np.squeeze(image, axis=0)
                    # print(np.max(image),np.min(image))
                    # break
                    image[image>1]=1
                    image[image<0]=0
                    # Normalize the pixel values to the range of 0 to 1

                    # Create a PIL Image object from the numpy array
                    image = Image.fromarray((image * 255).astype(np.uint8))
                
                else:
                    # Reshape the image from (3, 64, 64) to (64, 64, 3)
                    image = np.transpose(image, (1, 2, 0))
                    #print(np.max(image), np.min(image))
                    
                    # Normalize the pixel values to the range of 0 to 1
                    # image = image.astype(np.float32) / 255.0

                    # Create a PIL Image object from the numpy array
                    image = Image.fromarray((image * 255).astype(np.uint8))

                # Save the image with a unique filename
                filename = f"image_{i+1}.png"
                filepath = os.path.join(test_data_path, filename)
                image.save(filepath)

            print("Test Images saved successfully.")
        
        
        if args.vae:
            gen_data_path= './generated_image_folder/'+ args.dataset + '/' + args.task + '/vae' + '_d' +str(args.d)
        
        elif args.vanilla:
            gen_data_path= './generated_image_folder/'+ args.dataset + '/' + args.task + '/vanilla' + '_d' +str(args.d) + '_n'+str(args.n)
        
        elif args.irmae:
            gen_data_path= './generated_image_folder/'+ args.dataset + '/' + args.task + '/irmae_l' + str(args.l) + '_d' +str(args.d) + '_n' + str(args.n)
        
        else:
            gen_data_path= './generated_image_folder/'+ args.dataset + '/' + args.task + '/l' + str(args.l) + '_n' + str(args.n) + '_' + str(args.optimizer) + '_nuc_lambda_index' + str(args.nuc_lambda) + '_d' +str(args.d) 
        
        if not os.path.exists(gen_data_path):
            os.makedirs(gen_data_path)
            
            # Iterate over the stacked images and save each one individually
            for i, image in enumerate(y_hat):
                
                if args.dataset=='mnist' or args.dataset=='fmnist':

                    # Reshape the image from (1, 28, 28) to (28, 28)
                    image = np.squeeze(image, axis=0)
                    image[image>1]=1
                    image[image<0]=0

                    # Create a PIL Image object from the numpy array
                    image = Image.fromarray((image * 255).astype(np.uint8))

                    # Create a PIL Image object from the numpy array

                
                else:
                    # Reshape the image from (3, 64, 64) to (64, 64, 3)
                    image = np.transpose(image, (1, 2, 0))

                    # Create a PIL Image object from the numpy array
                    image = Image.fromarray((image * 255).astype(np.uint8))

                # Save the image with a unique filename
                filename = f"image_{i+1}.png"
                filepath = os.path.join(gen_data_path, filename)
                image.save(filepath)

            print("Generated Images saved successfully.")
        
        else:
            print('Gen img folder already exist, delete and re-run')
      
    
    else:
        # now plot
        for i in range(args.X):
            for j in range(args.Y):

                if args.dataset == 'mnist' or args.dataset== 'fmnist':
                    im = y_hat[i*args.Y+j][0, :, :]
                else:
                    im = np.transpose(y_hat[i*args.Y+j], [1, 2, 0])
                
                if args.dataset == 'mnist' or args.dataset=='fmnist':
                    axs[i, j].imshow(1-im, interpolation='nearest', cmap='Greys')
                else:
                    axs[i, j].imshow(im, interpolation='nearest')
                axs[i, j].axis('off')

        fig.tight_layout(pad=0.1)
        
        plt.savefig(path)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
