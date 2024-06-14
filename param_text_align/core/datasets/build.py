import os
import torch
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, ImageFolder, ImageNet


def build_dataset(cfg, train=True):
    # create a function that return a dataset based on the cfg.DATASETS.TRAIN
    # the dataset could be MNIST, CIFAR10, CIFAR100, Imagenette, ImageNet, etc.

    if cfg.DATASETS.TRAIN == "mnist":
        dataset = MNIST(
            root="datasets",
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )
        )
    elif cfg.DATASETS.TRAIN == "cifar10":
        dataset = CIFAR10(
            root="datasets",
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize((32, 32)), transforms.ToTensor()]
            ),
        )
    elif cfg.DATASETS.TRAIN == "cifar100":
        dataset = CIFAR100(
            root="datasets",
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize((32, 32)), transforms.ToTensor()]
            ),
        )
    
    elif cfg.DATASETS.TRAIN == "imagenette":
        split = "train" if cfg.DATASETS.TRAIN == "train" else "val"
        dataset = ImageFolder(
            root="datasets",
            split=split,
            download=True,
            size = "full",
            transform=transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor()]
            ),
        )
    
    elif cfg.DATASETS.TRAIN == "imagenet":
        split = "train" if cfg.DATASETS.TRAIN == "train" else "val"
        dataset = ImageNet(
            root="datasets",
            split=split,
            transform=transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor()]
            ),
        )

    else:
        raise ValueError(f"Unknown dataset: {cfg.DATASETS.TRAIN}")

    return dataset


def build_ae_dataset_flat(cfg, train=True):
    # the function should return a dataset that is used to train the autoencoder
    # each data sample should be a parameter of a model, that is loaded from the checkpoint in the result folder, where each folder name is the codification of the layers that compose the model.
    D_max = 65536 # (12800, 1024, '2_conv_256_512_5', '0_linear_1024_128')
    # N_max = 1024

    if train:
        data_path = cfg.DATASETS.TRAIN
    else:
        data_path = cfg.DATASETS.VALID

    dir_list = os.listdir(data_path)
    dataset_list = []
    for dir in dir_list:
        subdir_list = os.listdir(os.path.join(data_path, dir))
        for subdir in subdir_list:
            checkpoint = torch.load(os.path.join(data_path, dir, subdir))

            for k in checkpoint["state_dict"].keys():
                layer_name = k.split(".")[1]
                param_type = k.split(".")[2]

                arch_name = dir

                param = checkpoint["state_dict"][k]
                param = param.reshape(1, -1)
                N = param.shape[1]

                to_pad = D_max - param.shape[1]
                param = torch.cat((param.cpu().detach(), torch.zeros(1, to_pad)), dim=1)

                dataset_list.append((param, N, layer_name+'_'+param_type, arch_name))
    
    return ParamDataset(dataset_list)










def build_ae_dataset(cfg, train=True):
    # the function should return a dataset that is used to train the autoencoder
    # each data sample should be a parameter of a model, that is loaded from the checkpoint in the result folder, where each folder name is the codification of the layers that compose the model.
    D_max = 1024 # (12800, 1024, '2_conv_256_512_5', '0_linear_1024_128')
    # N_max = 1024

    if train:
        data_path = cfg.DATASETS.TRAIN
    else:
        data_path = cfg.DATASETS.VALID

    dir_list = os.listdir(data_path)
    dataset_list = []
    for dir in dir_list:
        subdir_list = os.listdir(os.path.join(data_path, dir))
        for subdir in subdir_list:
            checkpoint = torch.load(os.path.join(data_path, dir, subdir))

            for k in checkpoint["state_dict"].keys():
                layer_name = k.split(".")[1]
                param_type = k.split(".")[2]

                arch_name = dir

                # try:

                if param_type == "bias":
                    param = checkpoint["state_dict"][k]
                    param = param.reshape(1, -1)
                    to_pad = D_max - param.shape[1]
                    param = torch.cat((param.cpu().detach(), torch.zeros(1, to_pad)), dim=1)
                    N = param.shape[0]

                    # print(param.shape, N, layer_name, param_type)
                    dataset_list.append((param, N, layer_name+'_'+param_type, arch_name))

                elif 'batchnorm' in layer_name and (param_type == "weight" or param_type == "bias"):
                    param = checkpoint["state_dict"][k]
                    param = param.reshape(1, -1)
                    to_pad = D_max - param.shape[1]
                    param = torch.cat((param.cpu().detach(), torch.zeros(1, to_pad)), dim=1)
                    N = param.shape[0]

                    # print(param.shape, N, layer_name, param_type)
                    dataset_list.append((param, N, layer_name+'_'+param_type, arch_name))

                elif param_type == "weight":
                    param = checkpoint["state_dict"][k]
                    if 'conv' in layer_name:
                        # print(param.shape, layer_name)
                        # param = param.permute(1, 0, 2, 3)
                        param = param.reshape(param.shape[0], -1)
                        # print(param.shape)
                    # elif 'linear' in layer_name:
                    #     param = param.permute(1, 0)

                    to_pad = D_max - param.shape[1]
                    param = torch.cat((param.cpu().detach(), torch.zeros(param.shape[0], to_pad)), dim=1)
                    N = param.shape[0]
                
                    # print(param.shape, N, layer_name, param_type)
                    dataset_list.append((param, N, layer_name+'_'+param_type, arch_name))

                # except Exception as e:
                #     print(f"Error: {e}")
                #     print(f">>>>>>>>>>>>>>>>>>>>>>>>>>> Layer: {layer_name}, Param: {param_type}, shape: {param.shape}")
    
    return ParamDataset(dataset_list)


class ParamDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_list):
        self.dataset_list = dataset_list

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        return self.dataset_list[idx]
    






def check_max_dim(path='results/all_linear_pretrained/mnist'):
    # the function should return the maximum dimension of the data samples
    D_max = 0
    N_max = 0

    dir_list = os.listdir(path)
    for dir in dir_list:
        subdir_list = os.listdir(os.path.join(path, dir))
        for subdir in subdir_list:
            checkpoint = torch.load(os.path.join(path, dir, subdir))

            for k in checkpoint["state_dict"].keys():
                layer_name = k.split(".")[1]
                param_type = k.split(".")[2]

                param = checkpoint["state_dict"][k]
                param = param.reshape(1, -1)

                N = param.shape[0]
                D = param.shape[1]

                # if param_type == "bias" or ('batchnorm' in layer_name and (param_type == "weight" or param_type == "bias")):
                #     param = checkpoint["state_dict"][k]
                #     param = param.reshape(1, -1)
                #     N = param.shape[0]

                # elif param_type == "weight":
                #     param = checkpoint["state_dict"][k]
                #     if 'conv' in layer_name:
                #         param = param.permute(1, 0, 2, 3)
                #         param = param.reshape(param.shape[0], -1)
                #     elif 'linear' in layer_name:
                #         param = param.permute(1, 0)

                #     N = param.shape[0]

                if N > N_max:
                    N_max = N
                    N_name = layer_name

                if param.shape[1] > D_max:
                    D_max = param.shape[1]
                    D_name = layer_name

    return D_max, N_max, D_name, N_name





def list_of_parms(path='results/pretrained_models/mnist/conv_1_16_3_batchnorm2D_16_relu_avgpool_1_linear_16_10/_val_acc=0.239.ckpt'):
    D_max = 1024
    dataset_list = []

    data_path = '/'.join(path.split('/')[:3]) + '/'
    dir = path.split('/')[-2]
    subdir = path.split('/')[-1]

    checkpoint = torch.load(os.path.join(data_path, dir, subdir))

    for k in checkpoint["state_dict"].keys():
        layer_name = k.split(".")[1]
        param_type = k.split(".")[2]

        arch_name = dir

        try:

            if param_type == "bias":
                param = checkpoint["state_dict"][k]
                param = param.reshape(1, -1)
                to_pad = D_max - param.shape[1]
                param = torch.cat((param.cpu().detach(), torch.zeros(1, to_pad)), dim=1)
                N = param.shape[0]

                # print(param.shape, N, layer_name, param_type)
                dataset_list.append((param, N, layer_name+'.'+param_type, arch_name))

            elif 'batchnorm' in layer_name and (param_type == "weight" or param_type == "bias"):
                param = checkpoint["state_dict"][k]
                param = param.reshape(1, -1)
                to_pad = D_max - param.shape[1]
                param = torch.cat((param.cpu().detach(), torch.zeros(1, to_pad)), dim=1)
                N = param.shape[0]

                # print(param.shape, N, layer_name, param_type)
                dataset_list.append((param, N, layer_name+'.'+param_type, arch_name))

            elif param_type == "weight":
                param = checkpoint["state_dict"][k]
                if 'conv' in layer_name:
                    # print(param.shape, layer_name)
                    # param = param.permute(1, 0, 2, 3)
                    param = param.reshape(param.shape[0], -1)
                    # print(param.shape)
                # elif 'linear' in layer_name:
                #     param = param.permute(1, 0)

                to_pad = D_max - param.shape[1]
                param = torch.cat((param.cpu().detach(), torch.zeros(param.shape[0], to_pad)), dim=1)
                N = param.shape[0]
            
                # print(param.shape, N, layer_name, param_type)
                dataset_list.append((param, N, layer_name+'.'+param_type, arch_name))

        except Exception as e:
            print(f"Error: {e}")
            print(f">>>>>>>>>>>>>>>>>>>>>>>>>>> Layer: {layer_name}, Param: {param_type}, shape: {param.shape}")

    return dataset_list