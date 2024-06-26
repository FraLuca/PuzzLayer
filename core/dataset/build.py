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


    def __len__(self):
        return len(os.listdir(self.path))

    def __getitem__(self, idx):

        f = self.file_list[idx]
        rnd_ckpt_idx = torch.randint(0, self.max_num_ckpt, (1,)).item()
        modeltype = f[:4]

        if "MLP" in modeltype:
            data = torch.load(self.path + f)['pdata'][rnd_ckpt_idx]        
            data = partial_reverse_tomodel(data, self.model[modeltype])
        elif "CNN" in modeltype:
            data = torch.load(self.path + f, map_location="cpu")['pdata'][rnd_ckpt_idx]
        
        for param in data.parameters():
            param.requires_grad = False

        arch = sequential_to_arch(data)
        x, edge_index, edge_attr = arch_to_graph(arch)
        g_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        text = f.split('_')[2]
        text = text[1:-1] # remove from text "[", "]"
        text = text.replace(",", " ") # substitute "," with " "

        # text = f.split('_')[0] + ' ' + text

        return g_data, text, f


def custom_collate_fn(batch):
    data_list = [d[0] for d in batch]
    text_list = [d[1] for d in batch]
    f_list = [d[2] for d in batch]

    # for data in data_list:
    #     for key, value in data:
    #         if torch.is_tensor(value):
    #             value.requires_grad_(False)

    return Batch.from_data_list(data_list), text_list, f_list