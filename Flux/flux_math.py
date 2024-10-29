import torch
from einops import rearrange

def apply_rope(xq: torch.Tensor, 
               xk: torch.Tensor, 
               freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    
    xq_out = xq_out.reshape(*xq.shape).type_as(xq)
    xk_out = xk_out.reshape(*xk.shape).type_as(xk)
    
    return xq_out, xk_out

def attention(query:torch.Tensor, 
              key:torch.Tensor, 
              value:torch.Tensor, 
              positional_encoding:torch.Tensor) -> torch.Tensor:
    query, key = apply_rope(query, key, positional_encoding)

    x = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    x = rearrange(x, "B H L D -> B L (H D)")
    return x

def rope(position: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=position.device) / dim
    omega = 1.0 / (theta**scale)

    out = torch.einsum("...n,d->...nd", position, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)],
        dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2).float()
    return out

dim = 8
theta = 2
# [0, 2, 4, 6] -> [0, 1/4, 1/2, 3/4]
scale = torch.arange(0, dim, 2, dtype=torch.float64) / dim
# [2^0, 2^(1/4), 2^(1/2), 2^(3/4)] -> [1.0, 1.19, 1.41, 1.68] 
omega_division = (theta**scale)
# [1.00, 0.84, 0.70, 0.59]
omega = 1.0 / (theta**scale)

print(f"{dim=}")
print(f"{theta=}")
print(f"{scale=}")
print(f"{scale.shape=}")
print(f"{omega_division=}")
print(f"{omega_division.shape=}")
print(f"{omega=}")
print(f"{omega.shape=}")
