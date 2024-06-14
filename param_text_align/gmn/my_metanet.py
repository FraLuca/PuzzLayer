import torch
import torch.nn as nn
from graph_construct.model_arch_graph import sequential_to_arch, arch_to_graph, graph_to_arch, arch_to_sequential
from copy import deepcopy
from models.custom_model import *
from encoders import NodeEdgeFeatEncoder
from graph_models import EdgeMPNN
from graph_pooling import *
from torch_geometric.data import Data, Batch
import os

class MLP_mnist_4(nn.Module):
    def __init__(self, num_classes=10, input_channel=1):
        super(MLP_mnist_4, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(input_channel*28*28, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 25),
            nn.ReLU(),
            nn.Linear(25, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.module(x)
        return x
    
class MLP_mnist_2(nn.Module):
    def __init__(self, num_classes=10, input_channel=1):
        super(MLP_mnist_2, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(input_channel*28*28, 50),
            nn.ReLU(),
            nn.Linear(50, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.module(x)
        return x

def partial_reverse_tomodel(flattened, model):
    layer_idx = 0
    for name, pa in model.named_parameters():
        pa_shape = pa.shape
        pa_length = pa.view(-1).shape[0]
        pa.data = flattened[layer_idx:layer_idx + pa_length].reshape(pa_shape)
        pa.data.to(flattened.device)
        layer_idx += pa_length
    return model

# See torch version
print(torch.__version__)

# Load the model
model = torch.load("mnist/NND_mnist_run1.pt", map_location='cpu')['model'].module
model_4layer = MLP_mnist_4()
model_2layer = MLP_mnist_2()
print(model)

data_path = './toy_data/'
# Load the data
batch = []
for file in os.listdir(data_path):
    data = torch.load(data_path + file)['pdata'][0]
    data = partial_reverse_tomodel(data, model)
    arch = sequential_to_arch(data)
    x, edge_index, edge_attr = arch_to_graph(arch)
    g_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    batch.append(g_data)

batch = Batch.from_data_list(batch)
print("Batchified!")

# GNN pipeline definition
encoder = NodeEdgeFeatEncoder(64)
mpnn = EdgeMPNN(64, 64, 76, 20, 20, 3)
pooling = MLPEdgeReadout(20, 10, 10)
gnn = GNNwEdgeReadout(mpnn, pooling)

# check number of parameters of the gnn
# num_params = sum(p.numel() for p in gnn.parameters())
# print(f'Number of parameters of the GNN: {num_params}')
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

print('Success!')