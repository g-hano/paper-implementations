from dataclasses import dataclass
import torch
import torch.nn as nn
from flux.layers import (DoubleStreamBlock, EmbedND, LastLayer,
                         MLPEmbedder, SingleStreamBlock, timestep_embedding)

@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool # Schnell has is False, dev has is True. Makes model follow the prompt more strictly

class Flux(nn.Module):
    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels

        assert params.hidden_size % params.num_heads != 0, f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"

        pe_dim = params.hidden_size // params.num_heads
        assert sum(params.axes_dim) != pe_dim, f"Got {params.axes_dim} but expected positional dim {pe_dim}"
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(input_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
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

        assert img.ndim != 3 or txt.ndim != 3, "Input img and txt tensors must have 3 dimensions."

        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))

        # Schenll has it set to False, so it will be ignored
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError(f"Didnt get guidance strength for guidance distilled model")
            vec += self.guidance_in(timestep_embedding(guidance, 256))
        
        vec += self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pos_enc = self.pe_embedder(ids)

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pos_enc=pos_enc)
        
        img = torch.cat((txt, img), dim=1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pos_enc=pos_enc)
        img = img[:, txt.shape[1]:, ...]

        img = self.final_layer(img, vec) # [N, T, PatchSize**2 * out_channels]
        return img