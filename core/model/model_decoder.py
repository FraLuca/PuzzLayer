import torch

from core.model.utils.graph_utils.encoders import NodeEdgeFeatEncoder
from core.model.utils.graph_utils.graph_models import EdgeMPNN
from core.model.utils.graph_utils.graph_pooling import *
from core.configs import cfg
'''
class ModelDecoder(torch.nn.Module):
    def __init__(self, input_dim=1, output_dim=1):
        super(ModelDecoder, self).__init__()

        self.encoder = NodeEdgeFeatEncoder(64)
        self.mpnn = EdgeMPNN(64, 64, 76, 64, 64, 3, dropout=0.2)
        self.unpooling = MLPNodeEdgeUnpool(64, 64, 6)
        #self.gnn = GNNwEdgeReadout(mpnn, pooling)

    def forward(self, x, edge_index, batch, num_nodes):
        unpooled = self.unpooling(x, edge_index, batch, num_nodes)
        to_decode = self.mpnn(unpooled)

        encoded_x, encoded_edge = self.encoder(batch.x, batch.edge_attr)
        graph_encoding = self.gnn(encoded_x, batch.edge_index, encoded_edge, batch.batch)
        # if graph_encoding.shape[0] > 1: # if not sanity check
        #     # compute all pairwise differences between the rows of graph_encoding
        #     differences = torch.tensor([]).to(graph_encoding.device)
        #     for i in range(graph_encoding.shape[0]):
        #         for j in range(i+1, graph_encoding.shape[0]):
        #             diff = (graph_encoding[i] - graph_encoding[j]).abs().unsqueeze(0)
        #             differences = torch.cat((differences, diff), dim=0)
        #     print(differences.mean(dim=0))
        return graph_encoding
    





import torch
import torch.nn as nn

class BasicEdgeUnpool(nn.Module):
    def __init__(self, reduce='mean'):
        super().__init__()
        self.reduce = reduce

    def forward(self, graph_feat, edge_index, batch, num_nodes):
        if self.reduce == 'mean':
            count = torch.bincount(batch, minlength=num_nodes)
        elif self.reduce == 'sum':
            count = torch.ones_like(batch)
        else:
            raise ValueError(f"Unrecognized reduce method: {self.reduce}")

        # Broadcast graph_feat to the size of edge_index
        graph_feat_repeated = graph_feat[batch]

        return graph_feat_repeated / count.unsqueeze(-1)

class MLPNodeEdgeUnpool(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, reduce='mean'):
        super().__init__()
        self.unpool = BasicEdgeUnpool(reduce)
        layers = [nn.Linear(in_dim, hidden_dim)]
        layers.append(nn.ReLU())
        for _ in range(num_layers-2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, graph_feat, edge_index, batch, num_nodes):
        edge_attr_unpooled = self.unpool(graph_feat, edge_index, batch, num_nodes)
        return self.mlp(edge_attr_unpooled)'''


import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeUnpooler(nn.Module):
    def __init__(self):
        super(EdgeUnpooler, self).__init__()
        in_dim = 64
        hidden_dim = 64
        out_dim = 64
        num_layers = 2
        layers = [nn.Linear(in_dim, hidden_dim)]
        layers.append(nn.ReLU())
        for _ in range(num_layers-2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.mlp = nn.Sequential(*layers)
        

    def forward(self, graph_feat, batch):
        graph_feat = self.mlp(graph_feat)

        # Assuming batch is a 1D tensor with the same length as edge_attr and values between 0 and bs-1
        edge_batch = batch.batch[batch.edge_index[0]]

        # Unpool edge features
        edge_feat_unpooled = graph_feat[edge_batch]

        return edge_feat_unpooled

class ModelDecoder(nn.Module):
    def __init__(self):
        super(ModelDecoder, self).__init__()

        self.unpooling = EdgeUnpooler()
        self.mpnn = EdgeMPNN(64, 64, 76, 64, 64, 3, dropout=0.2)
        #self.decoder = NodeEdgeFeatDecoder(hidden_dim)
        
    def forward(self, graph_encoding, batch):
        # Unpooling
        edge_unpooled = self.unpooling(graph_encoding, batch)

        # GNN
        x, edge_attr = self.gnn(x, edge_index, edge_attr, None, batch)

        # Decoding
        #x_decoded, edge_attr_decoded = self.decoder(node_attr_unpooled, edge_attr_unpooled)

        return x_decoded, edge_attr_decoded


