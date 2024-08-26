import torch

from core.model.utils.graph_utils.encoders import NodeEdgeFeatEncoder
from core.model.utils.graph_utils.graph_models import EdgeMPNN
from core.model.utils.graph_utils.graph_pooling import *
from core.configs import cfg

class ModelEncoder(torch.nn.Module):
    def __init__(self, input_dim=1, output_dim=1, dropout=0.2):
        super(ModelEncoder, self).__init__()

        self.encoder = NodeEdgeFeatEncoder(hidden_dim=input_dim)
        mpnn = EdgeMPNN(node_in_dim=input_dim,
                        edge_in_dim=input_dim,
                        hidden_dim=76,
                        node_out_dim=64,
                        edge_out_dim=64,
                        num_layers=3,
                        dropout=dropout)
        pooling = MLPEdgeReadout(in_dim=64, hidden_dim=64, out_dim=output_dim)
        # pooling = DSNodeEdgeReadout(in_dim=64, hidden_dim=64, out_dim=output_dim)
        self.gnn = GNNwEdgeReadout(mpnn, pooling, use_nodes=False)

    def forward(self, batch, f=None):
        encoded_x, encoded_edge = self.encoder(batch.x, batch.edge_attr)
        graph_encoding = self.gnn(encoded_x, batch.edge_index, encoded_edge, batch.batch)

        # compute all pairwise differences between the rows of graph_encoding
        # differences = torch.tensor([]).to(graph_encoding.device)
        # for i in range(graph_encoding.shape[0]):
        #     for j in range(i+1, graph_encoding.shape[0]):
        #         diff = (graph_encoding[i] - graph_encoding[j]).abs().unsqueeze(0)
        #         differences = torch.cat((differences, diff), dim=0)
        # print(differences.mean(dim=0))
        # print(graph_encoding.std(dim=0)) # print std of graph_encodings
        return graph_encoding