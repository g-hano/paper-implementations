from dataclasses import dataclass
import torch
import torch.nn as nn
from einops import rearrange
from typing import List

def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

@dataclass
class AutoEncoderParams:
    resolution: int # 256
    in_channels: int # 3
    ch: int # 128
    out_ch: int # 3 
    ch_mult: List[int] # [1, 2, 4, 4] 
    num_res_blocks: int # 2
    z_channels: int # 16
    scale_factor: float # 0.3611 
    shift_factor: float # 0.1159

class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.q = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        """
    * What is GroupNorm? #! GroupNorm normalizes inputs by dividing channels into groups
    * Why we need GroupNorm? #! Works well with small batches, more stable training, independent of batch size
    * why we use Conv2d for query,key,value and not nn.Linear? #! Maintains spatial structure of image, more efficient than nn.Linear(). We would need to reshape->pass to linear->reshape
    * why we even have proj_out? What is the advantage? #! Mix information across channels after attention, another layer that can learn so it is useful
    * why we norm hidden_states in attention method? #! We want to stabilize the training and prevent attention scores from becoming too sharp, it helps learning
        """
    def attention(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        
        # [BatchSize, Channel, Height, Width]
        q = self.q(hidden_states)
        k = self.k(hidden_states)
        v = self.v(hidden_states)

        batch_size, channel, height, width = q.shape

        # [BatchSize, 1, Height*Width, Channel]
        # PyTorch scaled_dot_product_attention expects tensors in shape
        # [BatchSize, NumHeads, SeqLen, EmbedDim]
        # NumHeads: 1
        # SeqLen: Sequence of pixels=Height*Width
        # Dim: Channel

        #? We use rearrange because its easier to read
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()

        hidden_states = nn.functional.scaled_dot_product_attention(q, k, v)
        hidden_states = rearrange(hidden_states, 
                                  "b 1 (h w) c -> b c h w", 
                                  h=height, w=width)
        return hidden_states    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.attention(x)
        x = self.proj_out(x)
        return residual + x

#print("=====> Rearrange")
## [Batch, Channel, Height, Width]
#x = torch.randn(8, 3, 30, 30)
#print(f"{x.shape=}") # torch.Size([8, 3, 30, 30])

## [Batch, 1, Height*Width, Channel]
#x = rearrange(x, "b c h w -> b 1 (h w) c").contiguous()
#print(f"{x.shape=}") # torch.Size([8, 1, 900, 3])
#
#x = rearrange(x, "b 1 (h w) c -> b c h w", h=30, w=30)
#print(f"{x.shape=}") # torch.Size([8, 3, 30, 30])
#
#
#print("=====> Conv2d - Linear")
## [batch=1, channels=2, height=2, width=2]
#x = torch.tensor([[[[1, 2],[3, 4]],[[5, 6],[7, 8]]]], dtype=torch.float32)
#print(f"{x.shape=}") # [1, 2, 2, 2]
#
## 1. Using Conv2d with kernel_size=1
#conv = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1)
#with torch.no_grad():
#    conv.weight.data = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32).view(2, 2, 1, 1)
#    conv.bias.data = torch.tensor([0.0, 0.0], dtype=torch.float32)
#conv_output = conv(x)
#print(f"{conv_output.shape=}") # [1, 2, 2, 2]
#
## 2. Using Linear (requires reshaping)
#linear = nn.Linear(in_features=2, out_features=2)
#with torch.no_grad():
#    linear.weight.data = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
#    linear.bias.data = torch.tensor([0.0, 0.0], dtype=torch.float32)
#x_reshaped = rearrange(x, 'b c h w -> b (h w) c')
#print("Reshaped for Linear")
#print("Reshaped shape:", x_reshaped.shape)  # [1, 4, 2]
#print()
#linear_output = linear(x_reshaped)
#print("Linear output:")
#print(linear_output)
#print("Linear output shape:", linear_output.shape)  # [1, 4, 2]
#print()
#linear_output_reshaped = rearrange(linear_output, 'b (h w) c -> b c h w', h=2, w=2)
#print("Linear output reshaped back:")
#print("Final shape:", linear_output_reshaped.shape)  # [1, 2, 2, 2]

class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int | None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.norm1(hidden_states)
        hidden_states = (hidden_states * torch.sigmoid(hidden_states)) # swish(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states = (hidden_states * torch.sigmoid(hidden_states)) # swish(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.in_channels != self.out_channels:
            residual = self.nin_shortcut(residual)
        
        return residual + hidden_states
    
class DownSample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=in_channels, 
            kernel_size=3, stride=2, padding=0
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        asynmetric_padding = (0, 1, 0, 1)
        hidden_states = nn.functional.pad(hidden_states, 
                                          asynmetric_padding, 
                                          mode="constant", value=0)
        hidden_states = self.conv(hidden_states)
        return hidden_states
    
#ds = DownSample(3)
#x = torch.randn((2, 3, 16, 16))
#out = ds(x)
#print(f"{x.shape=}") # [2, 3, 16, 16] Batch,Channel,Height,Width
#print(f"{out.shape=}") # [2, 3, 8, 8]
#
#print(f"---------------------------")
#x = torch.randn((2, 3, 16, 16))
#print(f"{x.shape=}") # [1, 1, 1, 4]
## (1, 0, 1, 0)
## sol, sağ, üst, alt
#padx = nn.functional.pad(x, (0, 1, 0, 1), mode="constant", value=0)
#print(f"{padx.shape=}") # [1, 1, 2, 5]
#out = ds(x)
#print(f"{out.shape=}")

class UpSample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3, stride=1, padding=1
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = nn.functional.interpolate(hidden_states, 
                                                  scale_factor=2.0, mode="nearest") # double every value [1,2,3] ->[1,1,2,2,3,3] 
        hidden_states = self.conv(hidden_states)
        return hidden_states
    
#print(f"---------------------------")
#us = UpSample(3)
#x = torch.randn((2, 3, 16, 16))
#out = us(x)
#print(f"{x.shape=}") # [2, 3, 16, 16] Batch,Channel,Height,Width
#print(f"{out.shape=}") # [2, 3, 8, 8]

x = torch.rand((2, 3, 16, 16))
print(x.shape)
x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
print(x.shape)

x = torch.tensor([[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]], dtype=torch.float) # [1, 5, 2]
print(x.shape)
x = nn.functional.interpolate(x, scale_factor=3.0, mode="nearest")
print(x.shape)
print(x)

class Encoder(nn.Module):
    def __init__(self, 
                 resolution: int, 
                 in_channels: int, 
                 ch: int, ch_mult: List[int], 
                 num_res_blocks: int, z_channels: int):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # DownSampling
        self.conv_in = nn.Conv2d(in_channels=in_channels, 
                                 out_channels=self.ch, 
                                 kernel_size=3, stride=1, padding=1)

        current_res = resolution
        in_ch_mult = (1, ) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions): # 256
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks): #2
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = DownSample(block_in)
                current_res = current_res//2
            self.down.append(down)

        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # End
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2*z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # DownSampling
        hidden_states = [self.conv_in(hidden_states)]
        for i_level in range(self.num_resolutions): # 256
            for i_block in range(self.num_res_blocks): # 2
                hidden = self.down[i_level].block[i_block](hidden_states[-1])
                if len(self.down[i_level].attn) > 0:
                    hidden = self.down[i_level].attn[i_block](hidden)
                hidden_states.append(hidden)
            if i_level != self.num_resolutions-1:
                hidden_states.append(self.down[i_level].downsample(hidden_states[-1]))
        
        # Middle
        hidden = hidden_states[-1]
        hidden = self.mid.block_1(hidden)
        hidden = self.mid.attn_1(hidden)
        hidden = self.mid.block_2(hidden)

        # End
        hidden = self.norm_out(hidden)
        hidden = (hidden * torch.sigmoid(hidden)) # swish(hidden)
        hidden = self.conv_out(hidden)

        return hidden
    
class Decoder(nn.Module):
    def __init__(self,
                 ch: int,
                 out_ch: int,
                 ch_mult: List[int],
                 num_res_blocks: int,
                 in_channels: int,
                 resolution: int,
                 z_channels: int):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels # Bura kullanılmıyor hiç
        self.ffactor = 2 ** (self.num_resolutions - 1)

        # Compute in_ch_mult, block_in and current_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions-1]
        current_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1, z_channels, current_res, current_res)

        # z to block_in
        self.conv_in = nn.Conv2d(in_channels=z_channels, out_channels=block_in, 
                                 kernel_size=3, stride=1, padding=1)

        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # UpSampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = UpSample(block_in)
                current_res *= 2
            self.up.insert(0, up) # Prepend to get consistent order
        
        # End
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(in_channels=block_in, out_channels=out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # z to block_in
        hidden = self.conv_in(hidden_states)

        # Middle
        hidden = self.mid.block_1(hidden)
        hidden = self.mid.attn_1(hidden)
        hidden = self.mid.block_2(hidden)

        # UpSampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                hidden = self.up[i_level].block[i_block](hidden)
                if len(self.up[i_level].attn) > 0:
                    hidden = self.up[i_level].attn[i_block](hidden)
            if i_level != 0:
                hidden = self.up[i_level].upsample(hidden)
        
        # End
        hidden = self.norm_out(hidden)
        hidden = (hidden * torch.sigmoid(hidden)) #swish(hidden)
        hidden = self.conv_out(hidden)
        return hidden
    
class DiagonalGaussian(nn.Module):
    def __init__(self, sample: bool=True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        mean, logvar = torch.chunk(hidden_states, 2, dim=self.chunk_dim)
        if self.sample:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(mean)
        return mean
    
class AutoEncoder(nn.Module):
    def __init__(self, params: AutoEncoderParams):
        super.__init__()
        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.decoder = Decoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.reg = DiagonalGaussian()
        self.scale_factor = params.scale_factor
        self.shift_factor = params.shift_factor

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.reg(x)
        x = self.scale_factor * (x - self.shift_factor)
        return x
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = x / self.scale_factor + self.shift_factor
        x = self.decoder(x)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode(x)
        x = self.decode(x)
        return x