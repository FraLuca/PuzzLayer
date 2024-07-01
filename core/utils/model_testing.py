from core.model.utils.graph_construct.model_arch_graph import *
import torch
import os


def create_sequential_from_graph(graphs, original_sequentials):
    all_models = []
    for i in range(len(graphs)):
        arch = sequential_to_arch(original_sequentials[i])

        new_arch = graph_to_arch(arch, graphs[i].edge_attr[:,0])
        new_model = arch_to_sequential(new_arch, deepcopy(original_sequentials[i]))
        all_models.append(new_model)
    return all_models
        


