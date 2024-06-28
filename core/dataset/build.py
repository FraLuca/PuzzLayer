import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST #, CIFAR10, CIFAR100, ImageFolder, ImageNet
from torch_geometric.data import Data, Batch
import re

from core.model.utils.graph_construct.model_arch_graph import sequential_to_arch, arch_to_graph, partial_reverse_tomodel # , graph_to_arch, arch_to_sequential



def build_dataset(cfg, train=True):
    # create a function that return a dataset based on the cfg.DATASETS.TRAIN
    # the dataset could be MNIST, CIFAR10, CIFAR100, Imagenette, ImageNet, etc.

    dataset = ModelDataset(cfg, train=train)
    return dataset


class ModelDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, train=True):
        self.alignment = cfg.MODEL.ALIGNMENT
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
    
    def replace_conv2d_definitions(self, file_content):
        pattern = r'Conv2d\((\d+),\s*(\d+),\s*kernel_size=\((\d+),\s*(\d+)\),\s*stride=\((\d+),\s*(\d+)\)\)'
        replacement = r'Conv2d(in_channels=\1, out_channels=\2, kernel_size=(\3, \4), stride=(\5, \6))'
        modified_content = re.sub(pattern, replacement, file_content)
        return modified_content

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

        if self.alignment:
            classes = f.split('_')[2]
            classes = classes[1:-1] # remove from text "[", "]"
            classes = classes.replace(",", " ") # substitute "," with " "

            dataset_text = f.split('_')[1]

            sequential_text = ''
            for i, layer in enumerate(data):
                text_repr = layer.__repr__()
                if 'Conv2d' in text_repr:
                    text_repr = self.replace_conv2d_definitions(text_repr)
                sequential_text += text_repr
                if i != len(data) - 1:
                    sequential_text += ' -> '
                else:
                    sequential_text += ' [SEP] '

            
            # put the text in the format "sequential_text [SEP] dataset classes"
            text = sequential_text + dataset_text + ' ' + classes
        else:
            text = data

        # text = self.couples_to_onehot[text]

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