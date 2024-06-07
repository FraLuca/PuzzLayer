import torch

from core.model.utils.graph_utils.encoders import NodeEdgeFeatEncoder
from core.model.utils.graph_utils.graph_models import EdgeMPNN
from core.model.utils.graph_utils.graph_pooling import *
from core.configs import cfg

class ModelEncoder(torch.nn.Module):
    def __init__(self, input_dim=1, output_dim=1):
        super(ModelEncoder, self).__init__()

        self.encoder = NodeEdgeFeatEncoder(64)
        mpnn = EdgeMPNN(64, 64, 76, 20, 20, 3)
        pooling = MLPEdgeReadout(20, 10, cfg.MODEL.OUTPUT_DIM)
        self.gnn = GNNwEdgeReadout(mpnn, pooling)

    def forward(self, batch):
        encoded_x, encoded_edge = self.encoder(batch.x, batch.edge_attr)
        graph_encoding = self.gnn(encoded_x, batch.edge_index, encoded_edge, batch.batch)
        return graph_encoding