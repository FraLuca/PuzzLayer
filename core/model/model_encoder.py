import torch

from core.model.utils.graph_utils.encoders import NodeEdgeFeatEncoder
from core.model.utils.graph_utils.graph_models import EdgeMPNN
from core.model.utils.graph_utils.graph_pooling import *
from core.configs import cfg
from torch_geometric.nn import MetaLayer
from core.model.utils.graph_utils.graph_models import EdgeModel, NodeModel

class ModelEncoder(torch.nn.Module):
    def __init__(self, input_dim=1, output_dim=1):
        super(ModelEncoder, self).__init__()

        self.encoder = NodeEdgeFeatEncoder(64)
        mpnn = EdgeMPNN(64, 64, 76, 64, 64, 3, dropout=0.2)
        pooling = MLPEdgeReadout(64, 64, cfg.MODEL.OUTPUT_DIM)
        # pooling = MLPNodeEdgeReadout(128, 64, cfg.MODEL.OUTPUT_DIM)
        self.gnn = GNNwEdgeReadout(mpnn, pooling, use_nodes=False)

    def forward(self, batch, f=None):
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


class GraphUNetEncoder(torch.nn.Module):
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
        node_in_dim: int,
        edge_in_dim: int,
        pool_ratios: Union[float, List[float]] = 0.5,
        sum_res: bool = True,
        act: Union[str, Callable] = 'relu',
        reduce: str = 'mean',
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
        self.reduce = reduce

        channels = hidden_channels

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        # self.down_convs.append(GCNConv(in_channels, channels, improved=True))
        self.down_convs.append(MetaLayer(edge_model=EdgeModel(edge_in_dim+node_in_dim*2, channels),
                                    node_model=NodeModel(node_in_dim+channels, channels, reduce=self.reduce)))
        for i in range(depth):
            self.pools.append(TopKPooling(channels, self.pool_ratios[i]))
            # self.down_convs.append(GCNConv(channels, channels, improved=True))
            self.down_convs.append(MetaLayer(edge_model=EdgeModel(channels*3, channels),
                                    node_model=NodeModel(channels*2, channels, reduce=self.reduce)))

        in_channels = channels if sum_res else 2 * channels

        self.up_convs = torch.nn.ModuleList()
        for i in range(depth - 1):
            self.up_convs.append(MetaLayer(edge_model=EdgeModel(channels*3, channels),
                                    node_model=NodeModel(channels*2, channels, reduce=self.reduce)))
        self.up_convs.append(MetaLayer(edge_model=EdgeModel(channels*3, 1),
                                    node_model=NodeModel(channels*2, 3, reduce=self.reduce)))

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor,
                batch: OptTensor = None) -> Tensor:
        """"""  # noqa: D419
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        # edge_weight = x.new_ones(edge_index.size(1))

        x, edge_weight, _ = self.down_convs[0](x, edge_index, edge_weight)
        x = self.act(x)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.depth + 1):
            # edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
            #                                            x.size(0))
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
                x, edge_index, edge_weight, batch)

            x, edge_weight, _ = self.down_convs[i](x, edge_index, edge_weight)
            x = self.act(x)

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i

            # res = torch.randn_like(xs[j])
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            # up = torch.zeros_like(res)
            up = torch.randn_like(xs[j])
            up[perm] = x
            # set to zero rows of res where the corresponding row in up is not full zero
            # res[up.sum(dim=-1) != 0] = 0
            # x = res + up if self.sum_res else torch.cat((res, up), dim=-1)
            x = up

            x, edge_weight, _ = self.up_convs[i](x, edge_index, edge_weight)
            x = self.act(x) if i < self.depth - 1 else x

        return x, edge_weight

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
    

# if __name__ == "__main__":
    # create random graph and test forward pass of graph unet
    # from torch_geometric.data import Data
    # import torch
    # from torch_geometric.utils import to_undirected

    # # create random graph
    # edge_index = torch.tensor([])
    # edge_index = to_undirected(edge_index)
    # edge_attr = torch.randn(edge_index.size(1), 16)
    # x = torch.randn(25, 16)
    # data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    # data = data.to('cuda:6')

    # # create graph unet
    # model = GraphUNetEncoder(16, 32, 16, 3, 16, 16, 0.5)
    # model = model.to('cuda:6')

    # # forward pass
    # out = model(data.x, data.edge_index, data.edge_attr)
    # print(out.shape)
