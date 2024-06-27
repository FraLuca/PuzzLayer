import torch

from core.model.utils.graph_utils.encoders import NodeEdgeFeatEncoder
from core.model.utils.graph_utils.graph_models import EdgeMPNN
from core.model.utils.graph_utils.graph_pooling import *
from core.configs import cfg

class ModelEncoder(torch.nn.Module):
    def __init__(self, input_dim=1, output_dim=1):
        super(ModelEncoder, self).__init__()

        self.encoder = NodeEdgeFeatEncoder(64)
        mpnn = EdgeMPNN(64, 64, 76, 64, 64, 3, dropout=0.2)
        pooling = MLPEdgeReadout(64, 64, cfg.MODEL.OUTPUT_DIM)
        self.gnn = GNNwEdgeReadout(mpnn, pooling)

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