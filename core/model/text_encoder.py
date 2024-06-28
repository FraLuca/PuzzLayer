import torch
from core.configs import cfg
from transformers import BertTokenizer
from transformers import AutoTokenizer, AutoModel
from transformers import logging
import os
import torch.nn as nn
from core.model.utils.positional_encoding import PositionalEncoding

class TextEncoder(torch.nn.Module):
    def __init__(self, modelpath='bert-base-uncased', 
                        finetune=False,
                        latent_dim=64,
                        dropout=0.1,
                        ):
        super(TextEncoder, self).__init__()

        logging.set_verbosity_error()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        modelpath = 'bert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath)
        self.text_model = AutoModel.from_pretrained(modelpath)

        # Don't train the model
        if not finetune:
            self.text_model.training = False
            for p in self.text_model.parameters():
                p.requires_grad = False

        # Then configure the model
        self.text_encoded_dim = self.text_model.config.hidden_size

        # Projection of the text-outputs into the latent space
        self.projection = nn.Sequential(nn.ReLU(),
                                        nn.Linear(self.text_encoded_dim, latent_dim))

        self.emb_token = nn.Parameter(torch.randn(latent_dim))

        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout)

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                             nhead=4,
                                                             dim_feedforward=256,
                                                             dropout=dropout,
                                                             activation='gelu')
        
        self.seqTransEncoder = nn.TransformerEncoder(seq_trans_encoder_layer,
                                                     num_layers=4)



    def forward(self, texts):
        encoded_inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        # encoded inputs to model device
        encoded_inputs = {key: value.to(self.text_model.device) for key, value in encoded_inputs.items()}

        outputs = self.text_model(**encoded_inputs)
        text_encoded, mask = outputs.last_hidden_state, encoded_inputs['attention_mask'].to(dtype=bool)

        x = self.projection(text_encoded)
        bs, n_tokens, _ = x.shape
        x = x.permute(1, 0, 2)

        emb_token = torch.tile(self.emb_token, (bs,)).reshape(bs, -1)
        # adding the embedding token for all sequences
        xseq = torch.cat((emb_token[None], x), 0)

        # create a bigger mask, to allow attend to emb
        token_mask = torch.ones((bs, 1), dtype=bool, device=x.device)
        aug_mask = torch.cat((token_mask, mask), 1)

        # add positional encoding
        xseq = self.sequence_pos_encoding(xseq)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)

        return final[0]



class TextEncoderCustom(torch.nn.Module):
    def __init__(self, input_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        super(TextEncoderCustom, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        vocab_size = self.tokenizer.vocab_size

        self.max_length = 200

        self.embed_scale = input_dim ** 0.5
        self.embed_tokens = torch.nn.Embedding(vocab_size, input_dim)
        self.pos_embed = torch.nn.Parameter(torch.randn((1, self.max_length, input_dim)))
        # self.seg_embed = torch.nn.Parameter(torch.randn((2, input_dim)))
        self.dropout = torch.nn.Dropout(dropout)

        self.layers = torch.nn.ModuleList([])
        for i in range(num_layers):
            self.layers.append(
                torch.nn.TransformerEncoderLayer(
                    d_model=input_dim,
                    nhead=num_heads,
                    batch_first=True,
                )
            )

    def forward(self, x):
        x = self.tokenize_text_batch(x)['input_ids'].to(self.embed_tokens.weight.device)
        x = self.embed_tokens(x) * self.embed_scale
        x = x + self.pos_embed
        # x[:, :1] = x[:, :1] + self.seg_embed[0][None, None, :]
        # x[:, 1:] = x[:, 1:] + self.seg_embed[1][None, None, :]
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        return x.mean(dim=1)
        # return x[:,0]





    def tokenize_text_batch(self, text_batch):
        return self.tokenizer(
            text_batch,
            padding='max_length',   # Aggiungere padding se la sequenza è più corta della lunghezza fissa
            truncation=True,        # Troncare la sequenza se è più lunga della lunghezza fissa
            max_length=self.max_length,  # Lunghezza fissa desiderata
            return_tensors='pt'     # Restituire tensori PyTorch
        )