from dataclasses import dataclass
import torch
import torch.nn as nn
from layers import (DoubleStreamBlock, EmbedND, LastLayer,
                         MLPEmbedder, SingleStreamBlock, timestep_embedding)
from typing import List

@dataclass
class FluxParams:
    in_channels: int # 64
    out_channels: int # 64
    vec_in_dim: int # 768
    context_in_dim: int # 4096
    hidden_size: int # 3072
    mlp_ratio: float # 4.0
    num_heads: int # 24
    depth: int # 19
    depth_single_blocks: int # 38 
    axes_dim: List[int] # [16, 56, 56]
    theta: int # 10**4
    qkv_bias: bool # True
    guidance_embed: bool # Schnell has is False, dev has is True. Makes model follow the prompt more strictly

class Flux(nn.Module):
    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels

        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads

        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(input_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(input_dim=params.vec_in_dim, hidden_dim=self.hidden_size)
        # Schnell does not use guidance_embed, so it will have no effect at all.
        self.guidance_in = (
            MLPEmbedder(input_dim=256, hidden_dim=self.hidden_size) 
            if params.guidance_embed 
            else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size, 
                    self.num_heads, 
                    mlp_ratio=params.mlp_ratio,
                )
                for _ in range(params.depth_single_blocks)
            ]
        )
        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def forward(
        self, 
        img: torch.Tensor,
        img_ids: torch.Tensor,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: torch.Tensor,
        y: torch.Tensor,
        guidance: torch.Tensor | None = None) -> torch.Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        img = self.img_in(img)
        
        step_emb = timestep_embedding(timesteps, 256).to(torch.bfloat16)
        vec = self.time_in(step_emb)

        # Schnell has it set to False, so it will be ignored
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        
        vec = vec + self.vector_in(y)

        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pos_enc = self.pe_embedder(ids)

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pos_enc=pos_enc)
        
        img = torch.cat((txt, img), dim=1)

        for block in self.single_blocks:
            img = block(img, vec=vec, pos_enc=pos_enc)

        # img => (224, 224, 3) H, W, C
        # txt => (64, 64)
        img = img[:, txt.shape[1]:, ...]
        # now image contains information about the prompt as well because
        # we concatinated them and saved as `img`
        # so we slice `txt` part using txt.shape[1]
        # img will have a shape [224, 224-64, 3] = [224, 160, 3]

        img = self.final_layer(img, vec) # [N, T, PatchSize**2 * out_channels]
        return img