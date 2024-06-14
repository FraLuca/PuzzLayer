import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST #, CIFAR10, CIFAR100, ImageFolder, ImageNet
from torch_geometric.data import Data, Batch

from core.model.utils.graph_construct.model_arch_graph import sequential_to_arch, arch_to_graph, partial_reverse_tomodel # , graph_to_arch, arch_to_sequential



def build_dataset(cfg, train=True):
    # create a function that return a dataset based on the cfg.DATASETS.TRAIN
    # the dataset could be MNIST, CIFAR10, CIFAR100, Imagenette, ImageNet, etc.

    dataset = ModelDataset(cfg, train=train)
    return dataset


class ModelDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, train=True):
        if train:
            self.path = cfg.DATASETS.TRAIN
        else:
            self.path = cfg.DATASETS.TEST
        self.file_list = os.listdir(self.path)
        # self.max_num_ckpt = torch.load(self.path + self.file_list[0])['pdata'].shape[0]
        self.max_num_ckpt = 2

        # model = torch.load("mnist/NND_mnist_run1.pt", map_location='cpu')['model'].module  # TODO, we need to save module when we create data
        self.model = {
            "MLP3" : nn.Sequential(
                        nn.Linear(1*28*28, 50),
                        nn.ReLU(),
                        nn.Linear(50, 25),
                        nn.ReLU(),
                        nn.Linear(25, 10)
                    ),
            "MLP2" : nn.Sequential(
                        nn.Linear(784, 64),
                        nn.ReLU(),
                        nn.Linear(64, 10)
                    ),
            "MLP4" : nn.Sequential(
                        nn.Linear(784, 50),
                        nn.ReLU(),
                        nn.Linear(50, 25),
                        nn.ReLU(),
                        nn.Linear(25, 25),
                        nn.ReLU(),
                        nn.Linear(25, 10)
                    ),
        }
    
        self.couples_to_class = {
            "0 1": 0,
            "0 2": 1,
            "0 3": 2,
            "0 4": 3,
            "0 5": 4,
            "0 6": 5,
            "0 7": 6,
            "0 8": 7,
            "0 9": 8,
            "1 2": 9,
            "1 3": 10,
            "1 4": 11,
            "1 5": 12,
            "1 6": 13,
            "1 7": 14,
            "1 8": 15,
            "1 9": 16,
            "2 3": 17,
            "2 4": 18,
            "2 5": 19,
            "2 6": 20,
            "2 7": 21,
            "2 8": 22,
            "2 9": 23,
            "3 4": 24,
            "3 5": 25,
            "3 6": 26,
            "3 7": 27,
            "3 8": 28,
            "3 9": 29,
            "4 5": 30,
            "4 6": 31,
            "4 7": 32,
            "4 8": 33,
            "4 9": 34,
            "5 6": 35,
            "5 7": 36,
            "5 8": 37,
            "5 9": 38,
            "6 7": 39,
            "6 8": 40,
            "6 9": 41,
            "7 8": 42,
            "7 9": 43,
            "8 9": 44
        }

        self.couples_to_onehot = {
            "0 1": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            "0 2": [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            "0 3": [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            "0 4": [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            "0 5": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            "0 6": [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            "0 7": [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            "0 8": [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            "0 9": [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            "1 2": [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            "1 3": [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
            "1 4": [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
            "1 5": [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            "1 6": [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            "1 7": [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            "1 8": [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            "1 9": [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            "2 3": [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            "2 4": [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            "2 5": [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            "2 6": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            "2 7": [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            "2 8": [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            "2 9": [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            "3 4": [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            "3 5": [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
            "3 6": [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            "3 7": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            "3 8": [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            "3 9": [0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            "4 5": [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            "4 6": [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
            "4 7": [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            "4 8": [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            "4 9": [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            "5 6": [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            "5 7": [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
            "5 8": [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
            "5 9": [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            "6 7": [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            "6 8": [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
            "6 9": [0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            "7 8": [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            "7 9": [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            "8 9": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
        }

    def __len__(self):
        return len(os.listdir(self.path))

    def __getitem__(self, idx):

        f = self.file_list[idx]
        rnd_ckpt_idx = torch.randint(0, self.max_num_ckpt, (1,)).item()
        modeltype = f[:4]

        if "MLP" in modeltype:
            data = torch.load(self.path + f)['pdata'][rnd_ckpt_idx]        
            data = partial_reverse_tomodel(data, self.model[modeltype])
        else:
            data = torch.load(self.path + f, map_location="cpu")['pdata'][rnd_ckpt_idx]
        
        for param in data.parameters():
            param.requires_grad = False

        arch = sequential_to_arch(data)
        x, edge_index, edge_attr = arch_to_graph(arch)
        g_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        text = f.split('_')[2]
        text = text[1:-1] # remove from text "[", "]"
        text = text.replace(",", " ") # substitute "," with " "

        text = self.couples_to_onehot[text]

        return g_data, text, f


def custom_collate_fn(batch):
    data_list = [d[0] for d in batch]
    text_list = [d[1] for d in batch]
    f_list = [d[2] for d in batch]

    # for data in data_list:
    #     for key, value in data:
    #         if torch.is_tensor(value):
    #             value.requires_grad_(False)

    return Batch.from_data_list(data_list), torch.tensor(text_list), f_list