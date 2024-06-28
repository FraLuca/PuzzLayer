import torch

from core.model.utils.graph_utils.encoders import NodeEdgeFeatEncoder
from core.model.utils.graph_utils.graph_models import EdgeMPNN
from core.model.utils.graph_utils.graph_pooling import *
from core.configs import cfg
from core.model.utils.graph_construct.constants import NODE_TYPES, EDGE_TYPES
import torch.nn as nn
import torch.nn.functional as F

class Sin(nn.Module):
    def forward(self, x): return torch.sin(x)

class EdgeUnpooler(nn.Module):
    def __init__(self):
        super(EdgeUnpooler, self).__init__()
        
    def forward(self, graph_feat, batch):
        # Assuming batch is a 1D tensor with the same length as edge_attr and values between 0 and bs-1
        edge_batch = batch.batch[batch.edge_index[0]]
        # Unpool edge features
        edge_feat_unpooled = graph_feat[edge_batch]
        return edge_feat_unpooled

class NodeUnpooler(nn.Module):
    def __init__(self):
        super(NodeUnpooler, self).__init__()
        
    def forward(self, graph_feat, batch):
        # Assuming batch is a 1D tensor with the same length as x and values between 0 and bs-1
        node_batch = batch.batch
        # Unpool node features
        node_feat_unpooled = graph_feat[node_batch]
        return node_feat_unpooled
    
class NodeEdgeUnpooler(nn.Module):
    def __init__(self):
        super(NodeEdgeUnpooler, self).__init__()
        in_dim = 64
        hidden_dim = 64
        out_dim = 128
        num_layers = 2
        layers = [nn.Linear(in_dim, hidden_dim)]
        layers.append(nn.ReLU())
        for _ in range(num_layers-2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

        self.edge_unpooler = EdgeUnpooler()
        self.node_unpooler = NodeUnpooler()

    def forward(self, graph_feat, batch):
        graph_feat = self.mlp(graph_feat)
        node_feat, edge_feat = torch.chunk(graph_feat, 2, dim=-1)
        edge_attr = self.edge_unpooler(edge_feat, batch)
        x = self.node_unpooler(node_feat, batch)

        return x, batch.edge_index, edge_attr, batch


class NodeEdgeFeatDecoder(nn.Module):
    def __init__(self, hidden_dim, norms=False, post_activation=False, ff=False, ff_scale=3, use_conv=True):
        super().__init__()

        self.norms = norms
        self.post_activation = post_activation
        self.use_conv = use_conv

        # Node decoder
        self.x_proj = nn.Linear(hidden_dim, 3*hidden_dim)
        if norms:
            self.x_norm = nn.LayerNorm(3*hidden_dim)
        self.node_layer_encoder = nn.Sequential(nn.Linear(hidden_dim, 1), Sin())
        self.neuron_num_encoder = nn.Sequential(nn.Linear(hidden_dim,1), Sin())
        #self.node_type_encoder = nn.Embedding(hidden_dim, len(NODE_TYPES))
        self.node_type_encoder = nn.Linear(hidden_dim, 1)

        

        # Edge decoder
        edge_proj_dim = 4*hidden_dim if use_conv else 3*hidden_dim
        self.edge_attr_proj = nn.Linear(hidden_dim, edge_proj_dim)
        self.weight_encoder = nn.Sequential(nn.Linear(hidden_dim,1), Sin())
        self.edge_layer_encoder = nn.Sequential(nn.Linear(hidden_dim,1), Sin())
        if use_conv: self.conv_pos_encoder = nn.Sequential(nn.Linear(hidden_dim,3), Sin())
        #self.edge_type_encoder = nn.Embedding(hidden_dim, len(EDGE_TYPES))
        self.edge_type_encoder = nn.Linear(hidden_dim, 1)
        
        if norms:
            self.edge_attr_norm = nn.LayerNorm(edge_proj_dim)
        
        if post_activation:
            self.activation = nn.ReLU()

    def forward(self, x, edge_attr):
        x = x.float() # AP: added by me, otherwise it won't work, strange...
        x = self.x_proj(x)
        if self.norms:
            x = self.x_norm(x)
        x0, x1, x2 = torch.chunk(x, 3, dim=-1)
        x0 = self.node_layer_encoder(x0)
        x1 = self.neuron_num_encoder(x1)
        x2 = self.node_type_encoder(x2)
        x = torch.cat((x0, x1, x2), 1)

        edge_attr = self.edge_attr_proj(edge_attr)
        if self.norms:
            edge_attr = self.edge_attr_norm(edge_attr)
        
        if self.use_conv:
            e0, e1, e2, e3 = torch.chunk(edge_attr, 4, dim=-1)
            e0 = self.weight_encoder(e0)
            e1 = self.edge_layer_encoder(e1)
            e2 = self.edge_type_encoder(e2)
            e3 = self.conv_pos_encoder(e3)
            edge_attr = torch.cat((e0, e1, e2, e3), 1)
        else:
            e0, e1, e2 = torch.chunk(edge_attr, 3, dim=-1)
            e0 = self.weight_encoder(e0)
            e1 = self.edge_layer_encoder(e1)
            e2 = self.edge_type_encoder(e2)
            edge_attr = torch.cat((e0, e1, e2), 1)
        
        if self.post_activation:
            x = self.activation(x)
            edge_attr = self.activation(edge_attr)

        return x, edge_attr

class ModelDecoder(nn.Module):
    def __init__(self):
        super(ModelDecoder, self).__init__()

        self.unpooling = NodeEdgeUnpooler()
        self.mpnn = EdgeMPNN(64, 64, 76, 64, 64, 3, dropout=0.2)
        self.decoder = NodeEdgeFeatDecoder(64)
        
    def forward(self, graph_encoding, batch):
        # Unpooling
        x, edge_index, edge_attr, batch_graph = self.unpooling(graph_encoding, batch) 

        # GNN
        x, edge_attr = self.mpnn(x, batch.edge_index, edge_attr, None, batch)

        # Decoding
        x_decoded, edge_attr_decoded = self.decoder(x, edge_attr)

        return x_decoded, edge_attr_decoded


