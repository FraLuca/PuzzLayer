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
    
class EdgeUnpoolerOperation(nn.Module):
    def __init__(self):
        super(EdgeUnpoolerOperation, self).__init__()
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

        self.edge_unpooler = EdgeUnpooler()
        
    def forward(self, graph_feat, batch):
        graph_feat = self.mlp(graph_feat)
        edge_attr = self.edge_unpooler(graph_feat, batch)
        return batch.x, batch.edge_index, edge_attr, batch
    
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

        #self.unpooling = NodeEdgeUnpooler()
        self.unpooling = EdgeUnpoolerOperation()
        self.mpnn = EdgeMPNN(3, 64, 76, 64, 64, 3, dropout=0.2) #EdgeMPNN(64, 64, 76, 64, 64, 3, dropout=0.2)
        self.decoder = NodeEdgeFeatDecoder(64)
        
    def forward(self, graph_encoding, batch):
        # Unpooling
        x, edge_index, edge_attr, batch_graph = self.unpooling(graph_encoding, batch) 

        # GNN
        x, edge_attr = self.mpnn(x, batch.edge_index, edge_attr, None, batch)

        # Decoding
        x_decoded, edge_attr_decoded = self.decoder(x, edge_attr)

        return x_decoded, edge_attr_decoded


########################################################

class SimplerModelDecoder(nn.Module):
    def __init__(self):
        super(SimplerModelDecoder, self).__init__()

        self.mlp1 = nn.Linear(64, 320)
        self.mlp2 = nn.Linear(32, 160)
        self.mlp3 = nn.Linear(16, 80)
        self.mlp4 = nn.Linear(8, 30)

    def forward(self, graph_encoding, batch):

        edge_batch = batch.batch[batch.edge_index[0]]
        mask = torch.zeros((graph_encoding.shape[0], 30000)).to(graph_encoding.device)
        for i in range(graph_encoding.shape[0]):
            mask[i, :torch.sum(edge_batch==i)] = 1        

        # compute node features
        x = self.mlp1(graph_encoding).reshape(-1, 10, 32)
        x = self.mlp2(x).reshape(-1, 100, 16)
        x = self.mlp3(x).reshape(-1, 1000, 8)
        x = self.mlp4(x).reshape(-1, 30000)
        # x = x * mask
        return x[mask.bool()]

from typing import Callable, List, Union

import torch
from torch import Tensor

from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import OptTensor, PairTensor
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    to_torch_csr_tensor,
)
from torch_geometric.utils.repeat import repeat

class GraphUNetDecoder(torch.nn.Module):
    r"""The Graph U-Net model from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_ paper which implements a U-Net like
    architecture with graph pooling and unpooling operations.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        depth (int): The depth of the U-Net architecture.
        pool_ratios (float or [float], optional): Graph pooling ratio for each
            depth. (default: :obj:`0.5`)
        sum_res (bool, optional): If set to :obj:`False`, will use
            concatenation for integration of skip connections instead
            summation. (default: :obj:`True`)
        act (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.nn.functional.relu`)
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        depth: int,
        pool_ratios: Union[float, List[float]] = 0.5,
        sum_res: bool = True,
        act: Union[str, Callable] = 'relu',
    ):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.pool_ratios = repeat(pool_ratios, depth)
        self.act = activation_resolver(act)
        self.sum_res = sum_res

        channels = hidden_channels

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.down_convs.append(GCNConv(in_channels, channels, improved=True))
        for i in range(depth):
            self.pools.append(TopKPooling(channels, self.pool_ratios[i]))
            self.down_convs.append(GCNConv(channels, channels, improved=True))

        in_channels = channels if sum_res else 2 * channels

        self.up_convs = torch.nn.ModuleList()
        for i in range(depth - 1):
            self.up_convs.append(GCNConv(in_channels, channels, improved=True))
        self.up_convs.append(GCNConv(in_channels, out_channels, improved=True))

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor,
                batch: OptTensor = None) -> Tensor:
        """"""  # noqa: D419
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_weight = x.new_ones(edge_index.size(1))

        x = self.down_convs[0](x, edge_index, edge_weight)
        x = self.act(x)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.depth + 1):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
                                                       x.size(0))
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
                x, edge_index, edge_weight, batch)

            x = self.down_convs[i](x, edge_index, edge_weight)
            x = self.act(x)

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            x = self.up_convs[i](x, edge_index, edge_weight)
            x = self.act(x) if i < self.depth - 1 else x

        return x

    def augment_adj(self, edge_index: Tensor, edge_weight: Tensor,
                    num_nodes: int) -> PairTensor:
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        adj = to_torch_csr_tensor(edge_index, edge_weight,
                                  size=(num_nodes, num_nodes))
        adj = (adj @ adj).to_sparse_coo()
        edge_index, edge_weight = adj.indices(), adj.values()
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.hidden_channels}, {self.out_channels}, '
                f'depth={self.depth}, pool_ratios={self.pool_ratios})')