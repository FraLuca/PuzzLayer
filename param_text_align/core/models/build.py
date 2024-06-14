import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions.distribution import Distribution
import numpy as np


from functools import reduce
from typing import List, Optional, Union
import copy
import re


# from core.models.vae_utils.tools.embeddings import TimestepEmbedding, Timesteps
from core.models.vae_utils.architecture.position_encoding_layer import PositionalEncoding
from core.models.vae_utils.architecture.cross_attention import (
    SkipTransformerEncoder,
    SkipTransformerDecoder,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from core.models.vae_utils.architecture.position_encoding import build_position_encoding
from core.models.vae_utils.utils.temos_utils import lengths_to_mask
"""
vae

skip connection encoder 
skip connection decoder

mem for each decoder layer
"""

def extract_layers_from_path(cfg=None, path=None):
    if path is None:
        path = cfg.resume_model

    arch = path.split("/")[-2]

    # Define patterns for different parts of the string
    patterns = [
        r'conv_\d+_\d+_\d+',   # Matches 'conv_1_256_7'
        r'batchnorm2D_\d+',     # Matches 'batchnorm2D_256'
        r'batchnorm1D_\d+',     # Matches 'batchnorm1D_256'
        r'relu',                # Matches 'relu'
        r'avgpool_\d+',         # Matches 'avgpool_1'
        r'linear_\d+_\d+'       # Matches 'linear_512_10'
        r'maxpool_\d+'          # Matches 'maxpool_1'
    ]
    
    # Combine patterns into one regular expression
    pattern = '|'.join('(?:{})'.format(p) for p in patterns)
    
    # Find all matches in the input string
    matches = re.findall(pattern, arch)
    
    return matches



def build_model(cfg=None, layer_list=None):
    # create a model based on the cfg.MODEL.LAYERLIST
    # the layers could be linear_dimin_dimout, conv_dimin_dimout_kernel, batchnorm1D_dimin, batchnorm2D_dimin, dropout_p, relu, etc.
    rename = False

    if layer_list is None:
        layer_list = cfg.MODEL.LAYER_LIST
        num_layer = len(layer_list)
        # Check if 'conv' is in the layer list
        there_is_CNN = any("conv" in s for s in layer_list)

    model = nn.Sequential()

    for layer_num, layer in enumerate(layer_list):
        layer_type = layer.split("_")[0]

        if layer_type == "linear":
            if layer_num == 0:
                if rename:
                    model.add_module(str(layer_num)+"_flatten", nn.Flatten())
                else:
                    model.append(nn.Flatten())
            if there_is_CNN and layer_num == num_layer-1:
                if rename:
                    model.add_module(str(layer_num)+"_flatten", nn.Flatten())
                else:
                    model.append(nn.Flatten())

            dim_in = int(layer.split("_")[1])
            dim_out = int(layer.split("_")[2])
            if rename:
                model.add_module(str(layer_num)+"_"+layer, nn.Linear(dim_in, dim_out))
            else:
                model.append(nn.Linear(dim_in, dim_out))

        elif layer_type == "conv":
            dim_in = int(layer.split("_")[1])
            dim_out = int(layer.split("_")[2])
            kernel = int(layer.split("_")[3])
            if rename:
                model.add_module(str(layer_num)+"_"+layer, nn.Conv2d(dim_in, dim_out, kernel))
            else:
                model.append(nn.Conv2d(dim_in, dim_out, kernel))

        elif layer_type == "batchnorm1D":
            dim_in = int(layer.split("_")[1])
            if rename:
                model.add_module(str(layer_num)+"_"+layer, nn.BatchNorm1d(dim_in))
            else:
                model.append(nn.BatchNorm1d(dim_in))

        elif layer_type == "batchnorm2D":
            dim_in = int(layer.split("_")[1])
            if rename:
                model.add_module(str(layer_num)+"_"+layer, nn.BatchNorm2d(dim_in))
            else:
                model.append(nn.BatchNorm2d(dim_in))

        elif layer_type == "dropout":
            p = float(layer.split("_")[1])
            if rename:
                model.add_module(str(layer_num)+"_"+layer, nn.Dropout(p))
            else:
                model.append(nn.Dropout(p))

        elif layer_type == "relu":
            if rename:
                model.add_module(str(layer_num)+"_"+layer, nn.ReLU())
            else:
                model.append(nn.ReLU())

        elif layer_type == "avgpool":
            kernel = int(layer.split("_")[1])
            if rename:
                model.add_module(str(layer_num)+"_"+layer, nn.AdaptiveAvgPool2d(kernel))
                model.add_module(str(layer_num)+"_flatten", nn.Flatten())
            else:
                model.append(nn.AdaptiveAvgPool2d(kernel))
                model.append(nn.Flatten())
        elif layer_type == "maxpool":
            kernel = int(layer.split("_")[1])
            if rename:
                model.add_module(str(layer_num)+"_"+layer, nn.AdaptiveMaxPool2d(kernel))
                #model.add_module(str(layer_num)+"_flatten", nn.Flatten())
            else:
                model.append(nn.AdaptiveMaxPool2d(kernel))
                #model.append(nn.Flatten())

        elif layer_type == "adapool":
            kernel = int(layer.split("_")[1])
            if rename:
                model.add_module(str(layer_num)+"_"+layer, nn.AdaptiveMaxPool2d(kernel))
                #model.add_module(str(layer_num)+"_flatten", nn.Flatten())
            else:
                model.append(nn.AdaptiveMaxPool2d(kernel))
                #model.append(nn.Flatten())

        else:
            raise ValueError(f"Layer {layer} not recognized")

    return model


def init_model(cfg, model):
    # initialize the model based on the cfg.MODEL.INITIALIZATION
    # the initialization could be xavier, kaiming, etc.

    print(f"Initializing model with {cfg.MODEL.INITIALIZATION}")

    if cfg.MODEL.INITIALIZATION == "xavier":
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
    elif cfg.MODEL.INITIALIZATION == "kaiming":
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
    elif cfg.MODEL.INITIALIZATION == "normal_custom":
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                std = (cfg.MODEL.FACTOR * np.sqrt(2)) / np.sqrt(m.weight.shape[0])
                nn.init.zeros_(m.bias)
                nn.init.normal_(m.weight, mean=0, std=std)
            elif isinstance(m, nn.Linear):
                # if weight uses a normal distribution with std = sqrt(2)/sqrt(in_dim) else if bias uses zero initialization
                std = cfg.MODEL.INIT_FACTOR * (np.sqrt(2) / np.sqrt(m.weight.shape[1]))
                nn.init.zeros_(m.bias)
                nn.init.normal_(m.weight, mean=0, std=std)
    else:
        raise ValueError(f"Initialization {cfg.MODEL.INITIALIZATION} not recognized")

    return model



def build_ae_model(cfg):

    encoder = nn.Sequential(
        nn.Conv1d(1, 1024, kernel_size=1),
        nn.ReLU(),
        nn.Conv1d(1024, 256, kernel_size=1),
        nn.ReLU(),
        nn.Conv1d(256, 128, kernel_size=1),
        nn.ReLU(),
        nn.Conv1d(128, 64, kernel_size=1),
    )

    decoder = nn.Sequential(
        nn.Conv1d(64, 128, kernel_size=1),
        nn.ReLU(),
        nn.Conv1d(128, 256, kernel_size=1),
        nn.ReLU(),
        nn.Conv1d(256, 1024, kernel_size=1),
        nn.ReLU(),
        nn.Conv1d(1024, 1, kernel_size=1),
    )

    return encoder, decoder






class MldVae(nn.Module):

    def __init__(self,
                 nfeats: int,
                 latent_dim: list = [1, 256],
                 ff_size: int = 1024,
                 num_layers: int = 9,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 arch: str = "encoder_decoder",
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 position_embedding: str = "sine",
                 ablation=None,
                 **kwargs) -> None:

        super().__init__()

        self.latent_size = latent_dim[0]
        self.latent_dim = latent_dim[-1]
        input_feats = nfeats
        output_feats = nfeats
        self.arch = arch
        self.mlp_dist = False #ablation.MLP_DIST
        self.pe_type = "mld" #ablation.PE_TYPE

        if self.pe_type == "actor":
            self.query_pos_encoder = PositionalEncoding(
                self.latent_dim, dropout)
            self.query_pos_decoder = PositionalEncoding(
                self.latent_dim, dropout)
        elif self.pe_type == "mld":
            pass
            # self.query_pos_encoder = build_position_encoding(
            #     self.latent_dim, position_embedding=position_embedding)
            # self.query_pos_decoder = build_position_encoding(
            #     self.latent_dim, position_embedding=position_embedding)
        else:
            raise ValueError("Not Support PE type")

        encoder_layer = TransformerEncoderLayer(
            self.latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )
        encoder_norm = nn.LayerNorm(self.latent_dim)
        self.encoder = SkipTransformerEncoder(encoder_layer, num_layers,
                                              encoder_norm)

        if self.arch == "all_encoder":
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = SkipTransformerEncoder(encoder_layer, num_layers,
                                                  decoder_norm)
        elif self.arch == "encoder_decoder":
            decoder_layer = TransformerDecoderLayer(
                self.latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = SkipTransformerDecoder(decoder_layer, num_layers,
                                                  decoder_norm)
        else:
            raise ValueError("Not support architecture!")

        if self.mlp_dist:
            self.global_motion_token = nn.Parameter(
                torch.randn(self.latent_size, self.latent_dim))
            self.dist_layer = nn.Linear(self.latent_dim, 2 * self.latent_dim)
        else:
            self.global_motion_token = nn.Parameter(
                torch.randn(self.latent_size * 2, self.latent_dim))

        self.skel_embedding = nn.Linear(input_feats, self.latent_dim)
        self.final_layer = nn.Linear(self.latent_dim, output_feats)

    def forward(self, features: Tensor, lengths: Optional[List[int]] = None):
        # Temp
        # Todo
        # remove and test this function
        print("Should Not enter here")

        z, dist = self.encode(features, lengths)
        feats_rst = self.decode(z, lengths)
        return feats_rst, z, dist

    def encode(
            self,
            features: Tensor,
            lengths: Optional[List[int]] = None
    ) -> Union[Tensor, Distribution]:
        if lengths is None:
            lengths = [len(feature) for feature in features]

        device = features.device

        bs, nframes, nfeats = features.shape
        mask = lengths_to_mask(lengths, device)

        x = features
        # Embed each human poses into latent vectors
        x = self.skel_embedding(x)

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

        # Each batch has its own set of tokens
        dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))

        # create a bigger mask, to allow attend to emb
        dist_masks = torch.ones((bs, dist.shape[0]),
                                dtype=bool,
                                device=x.device)
        aug_mask = torch.cat((dist_masks, mask), 1)

        # adding the embedding token for all sequences
        xseq = torch.cat((dist, x), 0)

        if self.pe_type == "actor":
            # xseq = self.query_pos_encoder(xseq)
            dist = self.encoder(xseq,
                                src_key_padding_mask=~aug_mask)[:dist.shape[0]]
        elif self.pe_type == "mld":
            # xseq = self.query_pos_encoder(xseq)
            dist = self.encoder(xseq,
                                src_key_padding_mask=~aug_mask)[:dist.shape[0]]
            # query_pos = self.query_pos_encoder(xseq)
            # dist = self.encoder(xseq, pos=query_pos, src_key_padding_mask=~aug_mask)[
            #     : dist.shape[0]
            # ]

        # content distribution
        # self.latent_dim => 2*self.latent_dim
        if self.mlp_dist:
            tokens_dist = self.dist_layer(dist)
            mu = tokens_dist[:, :, :self.latent_dim]
            logvar = tokens_dist[:, :, self.latent_dim:]
        else:
            mu = dist[0:self.latent_size, ...]
            logvar = dist[self.latent_size:, ...]

        # resampling
        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample()
        return latent, dist

    def decode(self, z: Tensor, lengths: List[int]):
        mask = lengths_to_mask(lengths, z.device)
        bs, nframes = mask.shape

        queries = torch.zeros(nframes, bs, self.latent_dim, device=z.device)

        # todo
        # investigate the motion middle error!!!

        # Pass through the transformer decoder
        # with the latent vector for memory
        if self.arch == "all_encoder":
            xseq = torch.cat((z, queries), axis=0)
            z_mask = torch.ones((bs, self.latent_size),
                                dtype=bool,
                                device=z.device)
            augmask = torch.cat((z_mask, mask), axis=1)

            if self.pe_type == "actor":
                # xseq = self.query_pos_decoder(xseq)
                output = self.decoder(
                    xseq, src_key_padding_mask=~augmask)[z.shape[0]:]
            elif self.pe_type == "mld":
                # xseq = self.query_pos_decoder(xseq)
                output = self.decoder(
                    xseq, src_key_padding_mask=~augmask)[z.shape[0]:]
                # query_pos = self.query_pos_decoder(xseq)
                # output = self.decoder(
                #     xseq, pos=query_pos, src_key_padding_mask=~augmask
                # )[z.shape[0] :]

        elif self.arch == "encoder_decoder":
            if self.pe_type == "actor":
                # queries = self.query_pos_decoder(queries)
                output = self.decoder(tgt=queries,
                                      memory=z,
                                      tgt_key_padding_mask=~mask).squeeze(0)
            elif self.pe_type == "mld":
                # queries = self.query_pos_decoder(queries)
                # mem_pos = self.mem_pos_decoder(z)
                output = self.decoder(
                    tgt=queries,
                    memory=z,
                    tgt_key_padding_mask=~mask,
                    # query_pos=query_pos,
                    # pos=mem_pos,
                )#.squeeze(0)
                # query_pos = self.query_pos_decoder(queries)
                # # mem_pos = self.mem_pos_decoder(z)
                # output = self.decoder(
                #     tgt=queries,
                #     memory=z,
                #     tgt_key_padding_mask=~mask,
                #     query_pos=query_pos,
                #     # pos=mem_pos,
                # ).squeeze(0)

        output = self.final_layer(output)
        # zero for padded area
        output[~mask.T] = 0
        # Pytorch Transformer: [Sequence, Batch size, ...]
        feats = output.permute(1, 0, 2)
        return feats









class MldAe(nn.Module):

    def __init__(self,
                 nfeats: int,
                 latent_dim: list = [1, 256],
                 ff_size: int = 1024,
                 num_layers: int = 9,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 arch: str = "encoder_decoder",
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 position_embedding: str = "learned",
                 ablation=None,
                 **kwargs) -> None:

        super().__init__()

        self.latent_size = latent_dim[0]
        self.latent_dim = latent_dim[-1]
        input_feats = nfeats
        output_feats = nfeats
        self.arch = arch
        self.mlp_dist = False #ablation.MLP_DIST
        self.pe_type = "mld" #ablation.PE_TYPE

        if self.pe_type == "actor":
            self.query_pos_encoder = PositionalEncoding(
                self.latent_dim, dropout)
            self.query_pos_decoder = PositionalEncoding(
                self.latent_dim, dropout)
        elif self.pe_type == "mld":
            self.query_pos_encoder = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding)
            self.query_pos_decoder = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding)
        else:
            raise ValueError("Not Support PE type")

        encoder_layer = TransformerEncoderLayer(
            self.latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )
        encoder_norm = nn.LayerNorm(self.latent_dim)
        self.encoder = SkipTransformerEncoder(encoder_layer, num_layers,
                                              encoder_norm)

        if self.arch == "all_encoder":
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = SkipTransformerEncoder(encoder_layer, num_layers,
                                                  decoder_norm)
        elif self.arch == "encoder_decoder":
            decoder_layer = TransformerDecoderLayer(
                self.latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = SkipTransformerDecoder(decoder_layer, num_layers,
                                                  decoder_norm)
        else:
            raise ValueError("Not support architecture!")

        if self.mlp_dist:
            self.global_motion_token = nn.Parameter(
                torch.randn(self.latent_size, self.latent_dim))
            self.dist_layer = nn.Linear(self.latent_dim, 2 * self.latent_dim)
        else:
            self.global_motion_token = nn.Parameter(
                torch.randn(self.latent_size * 2, self.latent_dim))

        self.skel_embedding = nn.Linear(input_feats, self.latent_dim)
        self.final_layer = nn.Linear(self.latent_dim, output_feats)

    def forward(self, features: Tensor, lengths: Optional[List[int]] = None):
        # Temp
        # Todo
        # remove and test this function
        print("Should Not enter here")

        z, dist = self.encode(features, lengths)
        feats_rst = self.decode(z, lengths)
        return feats_rst, z, dist

    def encode(
            self,
            features: Tensor,
            lengths: Optional[List[int]] = None
    ) -> Union[Tensor, Distribution]:
        if lengths is None:
            lengths = [len(feature) for feature in features]

        device = features.device

        bs, nframes, nfeats = features.shape
        mask = lengths_to_mask(lengths, device)

        x = features
        # Embed each human poses into latent vectors
        x = self.skel_embedding(x)

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

        # Each batch has its own set of tokens
        #dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))

        # create a bigger mask, to allow attend to emb
        #dist_masks = torch.ones((bs, dist.shape[0]),
        #                        dtype=bool,
        #                        device=x.device)
        aug_mask = mask

        # adding the embedding token for all sequences
        xseq = x

        if self.pe_type == "actor":
            xseq = self.query_pos_encoder(xseq)
            latent = self.encoder(xseq,
                                src_key_padding_mask=~aug_mask)#[:dist.shape[0]]
        elif self.pe_type == "mld":
            xseq = self.query_pos_encoder(xseq)
            latent = self.encoder(xseq,
                                src_key_padding_mask=~aug_mask)#[:dist.shape[0]]
            # query_pos = self.query_pos_encoder(xseq)
            # dist = self.encoder(xseq, pos=query_pos, src_key_padding_mask=~aug_mask)[
            #     : dist.shape[0]
            # ]
        
        # content distribution
        # self.latent_dim => 2*self.latent_dim

        return latent

    def decode(self, z: Tensor, lengths: List[int]):
        mask = lengths_to_mask(lengths, z.device)
        bs, nframes = mask.shape

        queries = torch.zeros(nframes, bs, self.latent_dim, device=z.device)
        

        # todo
        # investigate the motion middle error!!!

        # Pass through the transformer decoder
        # with the latent vector for memory
        if self.arch == "all_encoder":
            xseq = torch.cat((z, queries), axis=0)
            z_mask = torch.ones((bs, self.latent_size),
                                dtype=bool,
                                device=z.device)
            augmask = torch.cat((z_mask, mask), axis=1)

            if self.pe_type == "actor":
                xseq = self.query_pos_decoder(xseq)
                output = self.decoder(
                    xseq, src_key_padding_mask=~augmask)[z.shape[0]:]
            elif self.pe_type == "mld":
                xseq = self.query_pos_decoder(xseq)
                output = self.decoder(
                    xseq, src_key_padding_mask=~augmask)[z.shape[0]:]
                # query_pos = self.query_pos_decoder(xseq)
                # output = self.decoder(
                #     xseq, pos=query_pos, src_key_padding_mask=~augmask
                # )[z.shape[0] :]

        elif self.arch == "encoder_decoder":
            if self.pe_type == "actor":
                queries = self.query_pos_decoder(queries)
                output = self.decoder(tgt=queries,
                                      memory=z,
                                      tgt_key_padding_mask=~mask).squeeze(0)
            elif self.pe_type == "mld":
                queries = self.query_pos_decoder(queries)
                # mem_pos = self.mem_pos_decoder(z)
                output = self.decoder(
                    tgt=queries,
                    memory=z,
                    tgt_key_padding_mask=~mask,
                    # query_pos=query_pos,
                    # pos=mem_pos,
                )#.squeeze(0)
                # query_pos = self.query_pos_decoder(queries)
                # # mem_pos = self.mem_pos_decoder(z)
                # output = self.decoder(
                #     tgt=queries,
                #     memory=z,
                #     tgt_key_padding_mask=~mask,
                #     query_pos=query_pos,
                #     # pos=mem_pos,
                # ).squeeze(0)

        output = self.final_layer(output)
        # zero for padded area
        output[~mask.T] = 0
        # Pytorch Transformer: [Sequence, Batch size, ...]
        feats = output.permute(1, 0, 2)
        return feats