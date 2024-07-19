from core.model.utils.graph_construct.model_arch_graph import *
import torch
import os
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm



def create_sequentials_from_graphs(graphs, original_sequentials):
    '''graphs is a batch of graph data, original_sequentials is the list of sequentials'''
    all_models = []
    for i in range(len(graphs)):
        arch = sequential_to_arch(original_sequentials[i])

        new_arch = graph_to_arch(arch, graphs[i].edge_attr[:,0])
        new_model = arch_to_sequential(new_arch, deepcopy(original_sequentials[i]))
        all_models.append(new_model)
    return all_models


def test_on_mnist(reco_model, orig_model, filenames, limit_to_first=0):

    # load MNIST dataset
    mnist_test = datasets.MNIST('data',
                                train=False,
                                download=True,
                                transform=transforms.Compose([
                                        transforms.Resize((28, 28)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                    ])
                                )
    
    if limit_to_first == 0:
        limit_to_first = len(reco_model)

    orig_accuracies_sum = 0
    reco_accuracies_sum = 0

    for i in tqdm(range(limit_to_first)):
        orig_model[i].eval()
        reco_model[i].eval()

        classes = filenames[i].split('_')[2]
        device = next(orig_model[i].parameters()).device

        indices = [i for i, (_, label) in enumerate(mnist_test) if str(label) in classes]
        mnist_subset = Subset(mnist_test, indices)
        test_loader = DataLoader(mnist_subset, batch_size=64, shuffle=False)

        orig_correct = 0
        reco_correct = 0

        for data, target in test_loader:
            data, target = data.to(device), target.to(device) # data.half().to(device), target.to(device)
            output = orig_model[i](data)
            orig_correct += output.argmax(dim=1).eq(target).sum().item()
            output = reco_model[i](data)
            reco_correct += output.argmax(dim=1).eq(target).sum().item()
        
        orig_accuracies_sum += orig_correct/len(mnist_subset)
        reco_accuracies_sum += reco_correct/len(mnist_subset)

    avg_orig_accuracy = orig_accuracies_sum / limit_to_first
    avg_reco_accuracy = reco_accuracies_sum / limit_to_first

    return avg_reco_accuracy, avg_orig_accuracy
