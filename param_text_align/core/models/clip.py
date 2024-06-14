
from transformers import BertTokenizer
import torch
from torch import nn
import numpy as np
from core.models.transformer import Transformer, positional_encoding
from core.models.tokenizer import custom_tokenizer



class CLIP(nn.Module):
    def __init__(self, transformer_width, transformer_layers=6, transformer_heads=8, pe_dim=32):
        super().__init__()

        # self.text_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.text_encoder = BertModel.from_pretrained("bert-base-uncased")

        # self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.text_projection = nn.Linear(1, transformer_width)
        self.pe_projection = nn.Linear(pe_dim, transformer_width)
        self.sentence_projection = nn.Linear(1, transformer_width)
        self.layer_causality_projection = nn.Linear(1, transformer_width)

        self.pe = positional_encoding(pe_dim, max_len=1024)

        self.text_encoder = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=None, # self.build_attention_mask()
        )


        # text_tokenizer.special_tokens_map

        # self.vae = VAE() # da caricare pretrained o from scratch
        self.param_encoder = self.vae.encoder

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.model = nn.Sequential(
            nn.Linear(784, 50, bias=True),
            nn.ReLU(),
            nn.Linear(50, 25, bias=True),
            nn.ReLU(),
            nn.Linear(25, 10, bias=True)
        )


    def forward(self, batch):

        param, text = batch  # text should be a list of texts

        # encoded_input = self.text_encoder(["Hello, my dog is cute", "Hello, my cat is cute"], return_tensors='pt', padding=True, truncation=True)
        # encoded_input = self.text_tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        # text = text_tokenizer.batch_decode(encoded_input['input_ids'])
        # text_features = self.text_encoder(**encoded_input)

        input_ids, sentence_encoding, layer_causality_encoding = custom_tokenizer(self.model, layer_id=0, text_task='image classification', text_dataset='MNIST')

        text_embed = self.text_projection(input_ids.unsqueeze(-1))
        pe_embed = self.pe_projection(self.pe)
        sentence_embed = self.sentence_projection(sentence_encoding.unsqueeze(-1))
        # layer_causality_embed = self.layer_causality_projection(layer_causality_encoding.unsqueeze(-1))
        tf_input = text_embed + pe_embed + sentence_embed
        text_features = self.text_encoder(tf_input)

        # text_features = self.text_encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)




        param_features = self.param_encoder(param)

        # compute similarity between the two embeddings

        # normalized features
        param_features = param_features / param_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * param_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text

