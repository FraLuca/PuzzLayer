import torch
from core.configs import cfg
from transformers import BertTokenizer

class TextEncoder(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=1, num_layers=2, dropout=0.1):
        super(TextEncoder, self).__init__()
        self.embed_scale = embed_dim ** 0.5
        self.embed_tokens = torch.nn.Embedding(vocab_size, embed_dim)
        self.embed_sentence = torch.nn.Embedding(3, embed_dim)
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = torch.nn.Dropout(dropout)

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.layers = torch.nn.ModuleList([])
        for i in range(num_layers):
            if i == num_layers - 1:
                embed_dim = cfg.MODEL.OUTPUT_DIM

            self.layers.append(
                torch.nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    batch_first=True,
                    dropout=dropout,
                    
                )
            )

    def forward(self, x):
        #sentence_embeddings = self.create_sentence_attention_mask(x['input_ids'])
        #sentence_embeddings = self.embed_sentence(sentence_embeddings)
        x = self.embed_tokens(x['input_ids']) * self.embed_scale
        #x = x + sentence_embeddings + self.pos_embed
        x = x + self.pos_embed
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        return x.mean(dim=1)
        # return x[:,0]
    
    def create_sentence_attention_mask(self, input_ids):
        attention_mask = torch.zeros_like(input_ids)
        for i, input_id in enumerate(input_ids):
            sep_indices = (input_id == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
            current_value = 0
            for j in range(len(input_id)):
                attention_mask[i, j] = current_value
                if j in sep_indices:
                    current_value += 1
        return attention_mask