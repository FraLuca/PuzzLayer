import torch
import torch.nn as nn
from graph_construct.model_arch_graph import sequential_to_arch, arch_to_graph, graph_to_arch, arch_to_sequential
from copy import deepcopy
from models.custom_model import *
from encoders import NodeEdgeFeatEncoder
from graph_models import EdgeMPNN
from graph_pooling import *
from torch_geometric.data import Data, Batch

model = torch.load("mnist/NND_mnist_run1.pt", map_location='cpu')['model'].module
model2 = torch.load("mnist/MLP_mnist_5_run1.pt", map_location='cpu')['model'].module

arch = sequential_to_arch(model)
x, edge_index, edge_attr = arch_to_graph(arch)
arch2 = sequential_to_arch(model2)
x2, edge_index2, edge_attr2 = arch_to_graph(arch2)

# batch everything
data1 = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
data2 = Data(x=x2, edge_index=edge_index2, edge_attr=edge_attr2)
batch = Batch.from_data_list([data1, data2])

# GNN pipeline definition
encoder = NodeEdgeFeatEncoder(64)
mpnn = EdgeMPNN(64, 64, 76, 20, 20, 3)
pooling = MLPEdgeReadout(20, 10, 10)
gnn = GNNwEdgeReadout(mpnn, pooling)

# check number of parameters of the gnn
# num_params = sum(p.numel() for p in gnn.parameters())
# print(f'Number of parameters of the GNN: {num_params}')

print(batch)
encoded_x, encoded_edge = encoder(batch.x, batch.edge_attr)
graph_encoding = gnn(encoded_x, batch.edge_index, encoded_edge, batch.batch)
print(graph_encoding.shape)

##################################################

# new_arch = graph_to_arch(arch, edge_attr[:, 0])
# new_model = arch_to_sequential(new_arch, deepcopy(model))

# # check state dicts are the same
# sd1, sd2 = model.state_dict(), new_model.state_dict()
# for k, v in sd1.items():
#     assert (v == sd2[k]).all()

print('success!')