import torch
import torch.nn as nn
from torch import nn

# from mld.models.operator import PositionalEncoding
# from mld.models.operator.position_encoding import build_position_encoding
from core.model.utils.graph_utils.encoders import NodeEdgeFeatEncoder
from core.model.utils.graph_utils.graph_models import EdgeMPNN
from core.configs import cfg
import math

class Denoiser(nn.Module):

    def __init__(self,
                 latent_dim: list = [64],
                 flip_sin_to_cos: bool = True,
                 freq_shift: int = 0,
                 text_encoded_dim: int = 64) -> None:

        super().__init__()

        self.encoder = NodeEdgeFeatEncoder(hidden_dim=cfg.MODEL.MODEL_INPUT_DIM)
        self.mpnn = EdgeMPNN(node_in_dim=cfg.MODEL.MODEL_INPUT_DIM,
                        edge_in_dim=cfg.MODEL.MODEL_INPUT_DIM,
                        hidden_dim=64,
                        node_out_dim=64,
                        edge_out_dim=64,
                        num_layers=5,
                        dropout=0.2)
        self.decoding_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.latent_dim = latent_dim[-1]
        self.text_encoded_dim = text_encoded_dim


        # text condition
        # project time from text_encoded_dim to latent_dim
        self.time_proj = Timesteps(text_encoded_dim, flip_sin_to_cos,
                                    freq_shift)
        # self.time_embedding = TimestepEmbedding(text_encoded_dim,
        #                                         self.latent_dim)


    def forward(self,
                sample,
                timestep,
                text_emb,
                layer_limits,
                **kwargs):

        # 1. time_embedding
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        # timesteps = timestep.expand(sample.shape[1]).clone()
        time_emb = self.time_proj(timestep)
        time_emb = time_emb.to(dtype=sample.edge_attr.dtype)
        # [1, bs, latent_dim] <= [bs, latent_dim]
        # time_emb = self.time_embedding(time_emb).unsqueeze(0)

        # 2. condition + time embedding
        conditioning = time_emb + text_emb
        if cfg.MODEL.DIFFUSION_PER_LAYER:
            # condition only the layers that we want to diffuse
            zeros = torch.zeros_like(conditioning)
            for start, end in layer_limits:
                zeros[start:end+1,:] = conditioning[start:end+1,:]
            conditioning = zeros

        # 3. encoder
        encoded_x, encoded_edge = self.encoder(sample.x, sample.edge_attr)
        # add conditioning
        encoded_edge = encoded_edge + conditioning

        # 4. mpnn denoising + linear
        transformed_x, transformed_edge_attr = self.mpnn(encoded_x, sample.edge_index, encoded_edge)
        weights_denoised = self.decoding_head(transformed_edge_attr)

        return (weights_denoised, )


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, channel: int, time_embed_dim: int, act_fn: str = "silu"):
        super().__init__()

        self.linear_1 = nn.Linear(channel, time_embed_dim)
        self.act = None
        if act_fn == "silu":
            self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample):
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)
        return sample


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        return t_emb