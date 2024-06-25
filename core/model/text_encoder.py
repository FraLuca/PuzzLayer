import torch
from core.configs import cfg

class TextEncoder(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        super(TextEncoder, self).__init__()
        self.embed_scale = embed_dim ** 0.5
        self.embed_tokens = torch.nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, 3, embed_dim))
        self.dropout = torch.nn.Dropout(dropout)

        self.layers = torch.nn.ModuleList([])
        for i in range(num_layers):
            if i == num_layers - 1:
                embed_dim = cfg.MODEL.OUTPUT_DIM

            self.layers.append(
                torch.nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    batch_first=True,
                )
            )

    def forward(self, x):
        x = self.embed_tokens(x) * self.embed_scale
        x = x + self.pos_embed
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        return x.mean(dim=1)
        # return x[:,0]