import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from einops import rearrange
from flux_math import attention, rope

class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        num_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(num_axes)],
            dim=-3
        )
        emb = emb.unsqueeze(1)
        return emb

def timestep_embedding(tensor_: torch.Tensor,
                       dim,
                       max_period=10**4,
                       time_factor: float = 10.0**3):
    """
    Creates sinusoidal timestep embeddings
    
    takes:
    tensor_: 1-D torch.Tensor of N indices, one per batch element.
    dim: the dimension of the output
    max_period: controls the minimum frequency of the embeddings.
    
    returns (N, D) torch.Tensor of positional embeddings."""

    tensor_ = time_factor * tensor_
    half = dim // 2
    neg_period_log = -math.log(max_period)
    normalized_numbers = torch.arange(0, half, dtype=torch.float32) / half

    freqs = torch.exp(neg_period_log * normalized_numbers).to(tensor_.device)
    #! Anlamadım
    args = tensor_[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    #  embedding    zeros     pos_embed
    # [1, 2, 3, 4]   [0]   [1, 2, 3, 4, 0]
    # [1, 2, 3, 4]   [0]   [1, 2, 3, 4, 0]
    # [1, 2, 3, 4] + [0] = [1, 2, 3, 4, 0]
    # [1, 2, 3, 4]   [0]   [1, 2, 3, 4, 0]
    # [1, 2, 3, 4]   [0]   [1, 2, 3, 4, 0]
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    
    if torch.is_floating_point(tensor_):
        embedding = embedding.to(tensor_)
    return embedding

print("================\n\n")
x = torch.rand((5, 5))
print(f"{x.shape=}")
print(x)
cat = torch.cat([x, x[:, :1]], dim=-1)
print(f"{cat.shape=}")
print(cat)

class MLPEmbedder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(input_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # -----    -------    -------  
        # |   |    |     |    |     |  
        # |   |    |     |    |     |   
        # | x | -> | in  | -> | out | 
        # |   |    | silu|    |     |     
        # -----    -------    -------    
        x = self.in_layer(x)
        x = self.silu(x)
        x = self.out_layer(x)
        return x

class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor):
        x_dtype = x.dtype
        x = x.float() # need float to calculate mean
        
        # square tensor to make them positive
        # calculate mean on last dim.
        mean = torch.mean(x**2, dim=-1, keepdim=True)
        
        rrms = torch.rsqrt(mean + 1e-6) # 1.0 / (mean + 1e-6)

        # normalize x and multiply by learnable parameter scale
        x = (x*rrms).to(dtype=x_dtype) * self.scale
        return x

print("=====> RMSNorm")

x = torch.tensor([
    [1, 2, 3, 4, 5], 
    [6, 7, 8, 9, 10]], dtype=torch.float)
print(f"{x.shape=}") # [2, 5]
print(f"{x=}")

mean = torch.mean(x, dim=-1, keepdim=True)
print(f"{mean.shape=}")
print(f"{mean=}")

sqrt = torch.sqrt(mean)
print(f"{sqrt.shape=}") # [2, 1]
print(f"{sqrt=}")

rsqrt = torch.rsqrt(mean)
print(f"{rsqrt.shape=}") # [2, 1]
print(f"{rsqrt=}")

print(f"{(1.0 / sqrt)=}")

print(f"{(x*rsqrt).shape=}") # [2, 5] * [2, 1] = [2, 5]

class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # [Batch, L, Dim]
        query = self.query_norm(query).to(value)
        key = self.key_norm(key).to(value)
        return query, key
    
class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor, pos_enc: torch.Tensor) -> torch.Tensor:
        # [BatchSize, L, K*NumHeads*HeadDim]
        qkv = self.qkv(x)

        # [K, BatchSize, NumHeads, L, HeadDim]
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        
        # [K, BatchSize, NumHeads, L, HeadDim]
        q, k = self.norm(q, k, v)

        # BatchSize, L (NumHeads*HeadDim)
        x = attention(q, k, v, pos_enc) # apply_rope changes the shape
        x = self.proj(x)
        return x
    
@dataclass
class ModulationOut:
    shift: torch.Tensor
    scale: torch.Tensor
    gate: torch.Tensor

class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        """silu -> linear -> resize -> chunk"""
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier*dim, bias=True)
    
    def forward(self, vec: torch.Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        silu = nn.functional.silu(vec)
        # [X, Y, Z]
        out = self.lin(silu)
        # [X, 1, Y, Z]
        out = out[:, None, :]
        out = out.chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None
        )

x = torch.tensor([[[1, 2], [3, 4]]])
y = x[:, None, :]
print(f"{x.shape=}")
print(f"{y.shape=}")

class DoubleStreamBlock(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 mlp_ratio: float,
                 qkv_bias: bool=False):
        super().__init__()
        mlp_hidden_dim = int(hidden_size*mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(hidden_size, num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True)
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(hidden_size, num_heads, qkv_bias=qkv_bias)
        
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True)
        )


    def forward(self, 
                img: torch.Tensor, 
                txt: torch.Tensor, 
                vec: torch.Tensor, 
                pos_enc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # Prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # Prepare text for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # Run attention
        # [K B H L D] cat on H
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn = attention(q, k, v, pos_enc)
        txt_attn = attn[:, :txt.shape[1]]
        img_attn = attn[:, txt.shape[1]:]

        # calculate the img bloks
        img += img_mod1.gate * self.img_attn.proj(img_attn)
        img += img_mod2.gate * self.img_mlp((1+img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt += txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt += txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)

        return img, txt 
    
class SingleStreamBlock(nn.Module):
    def __init__(self, 
                 hidden_size: int, 
                 num_heads: int, 
                 mlp_ratio: float = 4.0, 
                 qk_scale: float | None = None):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5 # 1/sqrt()

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size*3+self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)
    
    def forward(self, x: torch.Tensor, vec: torch.Tensor, pos_enc: torch.Tensor) -> torch.Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        ln1 = self.linear1(x_mod)
        qkv, mlp = torch.split(ln1, [3*self.hidden_size, self.mlp_hidden_dim], dim=-1)   

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        attn = attention(q, k, v, pos_enc)
        cat = torch.cat((attn, self.mlp_act(mlp)), 2)
        output = self.linear2(cat)
        return x + mod.gate * output

class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size*patch_size*out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2*hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x