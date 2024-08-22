from core.model.utils.graph_construct.model_arch_graph import *
import torch
import os
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
import copy



def create_sequentials_from_graphs(graphs, original_sequentials):
    '''graphs is a batch of graph data, original_sequentials is the list of sequentials'''
    all_models = []
    for i in range(len(graphs)):
        arch = sequential_to_arch(original_sequentials[i])

        new_arch = graph_to_arch(arch, graphs[i].edge_attr[:,0])
        new_model = arch_to_sequential(new_arch, deepcopy(original_sequentials[i]))
        all_models.append(new_model)
    return all_models


def test_on_mnist(reco_model, orig_model, filenames, limit_to_first=0, print_each_acc=False):

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
    test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)
    
    if limit_to_first == 0:
        limit_to_first = len(reco_model)

    orig_accuracies_sum = 0
    reco_accuracies_sum = 0

    for i in tqdm(range(limit_to_first)):
        orig_model[i].eval()
        reco_model[i].eval()
        device = next(orig_model[i].parameters()).device

        orig_correct = 0
        reco_correct = 0
        dictionary = {}

        for data, target in test_loader:
            data, target = data.to(device), target.to(device) # data.half().to(device), target.to(device)
            output = orig_model[i](data)
            orig_correct += output.argmax(dim=1).eq(target).sum().item()
            output = reco_model[i](data)
            if print_each_acc:
                # count how many times each value in output.argmax(dim=1) appears
                for val in output.argmax(dim=1).tolist():
                    if val in dictionary:
                        dictionary[val] += 1
                    else:
                        dictionary[val] = 1

            reco_correct += output.argmax(dim=1).eq(target).sum().item()
        
        orig_accuracies_sum += orig_correct/len(mnist_test)
        reco_accuracies_sum += reco_correct/len(mnist_test)
        if print_each_acc:
            print(f"  model: {filenames[i]}, orig_acc: {round(orig_correct/len(mnist_test), 3)}, reco_acc: {round(reco_correct/len(mnist_subset), 3)}, dict: {dictionary}")

    avg_orig_accuracy = orig_accuracies_sum / limit_to_first
    avg_reco_accuracy = reco_accuracies_sum / limit_to_first

    return avg_reco_accuracy, avg_orig_accuracy


def test_adding_noise(orig_model, filenames):
    print("TESTING ADDING NOISE TO THE WEIGHTS")

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
    
    for i in range(len(orig_model)):
        if "CNN4" not in filenames[i]:
            continue
        # print(orig_model[i])
        # exit()

        orig_model[i].eval()
        noised = copy.deepcopy(orig_model[i])

        # add noise to the weights
        for name, param in noised.named_parameters():
            if name.startswith("10."):
                param.data += torch.randn_like(param.data) * 3

        classes = filenames[i].split('_')[2]
        device = next(orig_model[i].parameters()).device

        indices = [i for i, (_, label) in enumerate(mnist_test) if str(label) in classes]
        mnist_subset = Subset(mnist_test, indices)
        test_loader = DataLoader(mnist_subset, batch_size=64, shuffle=False)

        orig_correct = 0
        noised_correct = 0
        dictionary = {}

        for data, target in test_loader:
            data, target = data.to(device), target.to(device) # data.half().to(device), target.to(device)
            output = orig_model[i](data)
            orig_correct += output.argmax(dim=1).eq(target).sum().item()
            output = noised(data)
            # count how many times each value in output.argmax(dim=1) appears
            for val in output.argmax(dim=1).tolist():
                if val in dictionary:
                    dictionary[val] += 1
                else:
                    dictionary[val] = 1

            noised_correct += output.argmax(dim=1).eq(target).sum().item()

        print(f"  model: {filenames[i]}, orig_acc: {round(orig_correct/len(mnist_subset), 3)}, noised_acc: {round(noised_correct/len(mnist_subset), 3)}, dict: {dictionary}")
    
    exit()
    return


def diff_per_layer(reco_model, orig_model, filenames):

    sum_diffs = {}
    networks = 0
    for i in tqdm(range(len(orig_model))):
        if "CNN4" not in filenames[i]:
            continue

        networks += 1
        orig_model[i].eval()
        reco_model[i].eval()

        for name, param in orig_model[i].named_parameters():
            # mse of parameters
            diff = torch.nn.functional.mse_loss(reco_model[i].state_dict()[name], param)
            if name in sum_diffs:
                sum_diffs[name] += diff.item()
            else:
                sum_diffs[name] = diff.item()

    for name, diff in sum_diffs.items():
        print(f"{name}: {round(diff/networks, 3)}")

    exit()
    return