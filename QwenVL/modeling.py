import torch
import torch.nn as nn
import torch.nn.functional  as F
from transformers.activations import ACT2FN
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from configs import Qwen2VLConfig, Qwen2VLVisionConfig
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.utils import is_torchdynamo_compiling
import math

class Qwen2VLOutput:
    loss = None
    logits = None
    past_key_values = None
    hidden_states = None
    attention = None
    rope_deltas = None

class Qwen2VLRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10**4,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Qwen2VLConfig = None
    ):
        super().__init__()
        self.rope_kwargs = {}
        if config is None:
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings
        
        self.config = config
        #! Buna bak
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def update(self, position_ids, device):
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len
        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len
    
    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self.update(position_ids, device=x.device)

        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        # [3, BatchSize, 1, Positions]
        position_ids_expanded = position_ids[:, :, None, :].float()
        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        
        cos *= self.attention_scaling
        sin *= self.attention_scaling
        cos, sin = cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        print(f"{cos.dtype=}, {sin.dtype=}")
        return cos, sin
    
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    cat = torch.cat((-x2, x1), dim=-1)
    print(f"{cat.dtype=}")
    return cat

def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    print(f"{q_embed.dtype=}, {k_embed.dtype=}")
    return q_embed, k_embed

def apply_rotary_pos_emb_vision(tensor, freqs):
    orig_dtype = tensor.dtype
    tensor = tensor.float()

    cos = freqs.cos()
    sin = freqs.sin()

    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()

    output = (tensor * cos) + (rotate_half(tensor) * sin)
    output = output.to(orig_dtype)
    print(f"{output.dtype=}")
    return output

class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs
    
    def forward(self, seq_len):
        seq = torch.arange(seq_len, 
                           device=self.inv_freq.device, 
                           dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        print("--VisionRotaryEmbedding--")
        print(f"{freqs.dtype=}")
        return freqs

class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size = 14,
        temporal_patch_size = 2,
        in_channels = 3,
        embed_dim = 1152            
    ):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, 
                              embed_dim, 
                              kernel_size=kernel_size, 
                              stride=kernel_size, 
                              bias=False)

    def forward(self, hidden_states):
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        print("--PatchEmbed--")
        print(f"{hidden_states.dtype=}")
        return hidden_states

class PatchMerger(nn.Module):
    def __init__(self,
                 dim, 
                 context_dim, 
                 spatial_merge_size = 2):
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = nn.LayerNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim)
        )
    
    def forward(self, x):
        x = self.ln_q(x)
        x = x.view(-1, self.hidden_size)
        x = self.mlp(x)
        print("--PatchMerger--")
        print(f"{x.dtype=}")
        return x

class VisionMLP(nn.Module):
    def __init__(self, dim, hidden_dim, hidden_act):
        
        #              -----------
        # ---------    |         |                ---------
        # |       |    |         |    -------     |       |
        # |   x   | -> |   fc1   | -> | act |  -> |  fc2  |
        # |       |    |         |    -------     |       |
        # ---------    |         |                ---------
        #              -----------

        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = ACT2FN[hidden_act]
        self.fc2 = nn.Linear(hidden_dim, dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        print("--VisionMLP--")
        print(f"{x.dtype=}")
        return x

class VisionAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor = None
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        print(f"--VisionAttention--")
        print(f"{q.dtype=}")
        print(f"{k.dtype=}")
        print(f"{v.dtype=}") 

        attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)

        print(f"{attn_output.dtype=}")
        attn_output = self.proj(attn_output)
        print(f"{attn_output.dtype=}")
        return attn_output
    
    
class Qwen2VLVisionBlock(nn.Module):
    def __init__(self, config: Qwen2VLVisionConfig):
        super().__init__()
        self.eps = 1e-6
        self.norm1 = nn.LayerNorm(config.embed_dim, eps=self.eps)
        self.norm2 = nn.LayerNorm(config.embed_dim, eps=self.eps)

        mlp_hidden_dim = int(config.embed_dim * config.mlp_ratio)
        self.mlp = VisionMLP(dim=config.embed_dim, 
                             hidden_dim=mlp_hidden_dim, 
                             hidden_act=config.hidden_act)
        
        self.attn = VisionAttention(config.embed_dim, num_heads=config.num_heads)

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        print("--Qwen2VLVisionBlock--")
        print(f"{hidden_states.dtype=}")
        return hidden_states
    
# https://imgur.com/a/xe4HB54


class Cache(nn.Module):
    def __init__(self, num_hidden_layers=None) -> None:
        super().__init__()
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.key_cache = []
        self.value_cache = []

    def __getitem__(self, layer_idx: int):
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        return len(self.key_cache)
    
    def update(
        self,
        key_states,
        value_states,
        layer_idx,
        cache_kwargs=None,
    ):
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if len(self.key_cache) <= layer_idx:
            # There may be skipped layers, fill them with empty lists
            for _ in range(len(self.key_cache), layer_idx):
                self.key_cache.append([])
                self.value_cache.append([])
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        elif len(self.key_cache[layer_idx]) == 0:  # fills previously skipped layers; checking for tensor causes errors
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]    

    def get_seq_len(self, layer_idx):
        is_empty_layer = (
            len(self.key_cache) == 0  # no cache in any layer
            or len(self.key_cache) <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
            or len(self.key_cache[layer_idx]) == 0  # the layer has no cache
        )
        layer_seq_length = self.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        return layer_seq_length
    
    
    def batch_repeat_interleave(self, repeats: int):
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(repeats, dim=0)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor):
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][indices, ...]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][indices, ...]
    
    def to_legacy_cache(self):

        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache
    def get_max_cache_shape(self):
        return None

    def get_max_len(self, layer_idx):
        return self.max_cache_len
    
    def reset(self):
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx].zero_()
            self.value_cache[layer_idx].zero_()

    def get_usable_len(self, new_seq_len, layer_idx):
        max_len = self.get_max_len()
        previous_seq_len = self.get_seq_len(layer_idx)
        if max_len is not None and previous_seq_len + new_seq_len > max_len:
            return max_len - new_seq_len
        return previous_seq_len
    
    def reorder_cache(self, beam_idx):
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] != []:
                device = self.key_cache[layer_idx].device
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            if self.value_cache[layer_idx] != []:
                device = self.value_cache[layer_idx].device
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        print("--Qwen2RMSNorm--")
        print(f"{hidden_states.dtype=}")
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
     
class Qwen2MLP(nn.Module):
    def __init__(self, config: Qwen2VLConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states):
        x = self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        print(f"--Qwen2MLP--")
        print(f"{x.shape=} {x.dtype=}")
        return x

def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    hidden_states = hidden_states.reshape(batch, num_key_value_heads*n_rep, seq_len, head_dim)
    print(f"--repeat_kv--")
    print(f"{hidden_states.dtype=}")
    return hidden_states

class Qwen2VLAttention(nn.Module):
    def __init__(self, config: Qwen2VLConfig, layer_idx=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_scaling = config.rope_scaling
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Qwen2VLRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta
        )

    def forward(self, hidden_states, attn_mask=None, pos_ids=None, 
                past_kv=None, cache_pos=None, pos_emb=None):
        batch_size, seq_len, _ = hidden_states.size()

        q_states = self.q_proj(hidden_states)
        k_states = self.k_proj(hidden_states)
        v_states = self.v_proj(hidden_states)

        q_states = q_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_states = k_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v_states = v_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = k_states.shape[-2]
        if past_kv is not None:
            kv_seq_len += past_kv.get_usable_length(kv_seq_len, self.layer_idx)

        if pos_emb is None:
            cos, sin = self.rotary_emb(v_states, pos_ids)
        else:
            cos, sin = pos_emb

        q_states, k_states = apply_multimodal_rotary_pos_emb(
            q_states, k_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_kv is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_pos}
            k_states, v_states = past_kv.update(k_states, v_states, 
                                                self.layer_idx, cache_kwargs)
    
        # Repeat KV 
        k_states = repeat_kv(k_states, self.num_key_value_groups)
        v_states = repeat_kv(v_states, self.num_key_value_groups)
        causal_mask = attn_mask
        if attn_mask is not None:  # no matter the length, we just slice it
            causal_mask = attn_mask[:, :, :, : k_states.shape[-2]]

        if q_states.device.type == "cuda" and attn_mask is not None:
            q_states = q_states.contiguous()
            k_states = k_states.contiguous()
            v_states = v_states.contiguous()

        is_causal = True if causal_mask is None and seq_len > 1 else False

        attn_output = nn.functional.scaled_dot_product_attention(
            q_states,
            k_states,
            v_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal
        )
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)
        print("--Qwen2VLAttention--")
        print(f"{attn_output.dtype=}")
        return attn_output, None, past_kv
    
class Qwen2VLDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2VLConfig, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2VLAttention(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.pos_attn_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, 
                hidden_states, attn_mask=None, 
                pos_ids=None, past_kv=None, 
                output_attn=None, use_cache=None, 
                cache_pos=None, pos_emb=None):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention 
        hidden_states, self_attn_weights, present_kv = self.self_attn(
            hidden_states=hidden_states,
            attn_mask=attn_mask,
            pos_ids=pos_ids,
            past_kv=past_kv,
            cache_pos=cache_pos,
            pos_emb=pos_emb
        )
        hidden_states += residual

        # Fully Connected
        residual = hidden_states
        hidden_states = self.pos_attn_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states += residual
        print("--Qwen2VLDecoderLayer--")
        print(f"{hidden_states.dtype=}")

        outputs = (hidden_states, )
        
        if output_attn:
            outputs += (self_attn_weights, )
        
        if use_cache:
            outputs += (present_kv, )
        
        return outputs
    
class Qwen2VLPreTrainedModel(nn.Module):
    def __init__(self, config, generation_config):
        super().__init__()
        self.config = config
        self.generation_config = generation_config

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=self.config.initializer_range)
            elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def post_init(self):
        self.init_weights()

class Qwen2VisionTransformerPretrainedModel(Qwen2VLPreTrainedModel):
    def __init__(self, config: Qwen2VLVisionConfig, generation_config):
        super().__init__(config, generation_config)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim
        )

        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [Qwen2VLVisionBlock(config) for _ in range(config.depth)]
        )
        self.merger = PatchMerger(
            dim=config.hidden_size, context_dim=config.embed_dim, 
            spatial_merge_size=config.spatial_merge_size
        )

    def get_dtype(self):
        return self.blocks[0].mlp.fc2.weight.dtype
    
    def rotate_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()

            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb
    
    def forward(self, hidden_states, grid_thw):
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rotate_pos_emb(grid_thw)

        cu_seqlen = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlen = F.pad(cu_seqlen, (1, 0), value=0)

        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens=cu_seqlen, 
                                  rotary_pos_emb=rotary_pos_emb)
        return self.merger(hidden_states)

class Qwen2VLModel(Qwen2VLPreTrainedModel):
    def __init__(self, config: Qwen2VLConfig, generation_config):
        super().__init__(config, generation_config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, 
                                         config.hidden_size, 
                                         self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2VLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2VLRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, 
                                          past_seen_tokens+inputs_embeds.shape[1], 
                                          device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # Create positional embeddings to be shared across the decoders
        positional_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Decoder Layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attn = () if output_hidden_states else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states, )
            
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    positional_embeddings
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attn_mask=causal_mask,
                    pos_ids=position_ids,
                    past_kv=past_key_values,
                    output_attn=output_attentions,
                    use_cache=use_cache,
                    cache_pos=cache_position,
                    pos_emb=positional_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            
            if output_attentions:
                all_self_attn += (layer_outputs[1], )

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states, )
        
        next_cache = next_decoder_cache if use_cache else None

        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attn] if v is not None)
    
    def _update_causal_mask(
        self, attn_mask, input_tensor,
        cache_pos, past_kv, output_att   
    ):
        past_seen_tokens = past_kv.get_seq_length() if past_kv is not None else 0

        dtype, device = input_tensor.dtype, input_tensor.device
        seq_len = input_tensor.shape[1]

        target_len = (attn_mask.shape[-1] if isinstance(attn_mask, torch.Tensor) else past_seen_tokens+seq_len+1)

        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attn_mask, seq_len=seq_len, target_len=target_len,
            dtype=dtype, device=device, cache_pos=cache_pos,
            batch_size=input_tensor.shape[0], config=self.config, past_kv=past_kv
        )

        if (self.config._attn_implementation == "sdpa"
            and attn_mask is not None
            and attn_mask.device.type == "cuda"
            and not output_att):
            min_dtype = torch.finfo(dtype).min
            #!!!!!!!!!!!!!!
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
        
        return causal_mask
    
    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attn_mask, seq_len,
        target_len, dtype,
        device, cache_pos, 
        batch_size, config, past_kv
    ):
        if attn_mask is not None and attn_mask.dim() == 4:
            causal_mask = attn_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (seq_len, target_len), fill_value=min_dtype, 
                dtype=dtype, device=device
            )
            diagonal_attend_mask = torch.arange(target_len, device=device) > cache_pos.reshape(-1, 1)
            if config.sliding_window is not None:
                if seq_len > target_len:
                    sliding_attend_mask = torch.arange(target_len, device=device) <= (
                        cache_pos.reshape(-1, 1) - config.sliding_window
                    )
                    diagonal_attend_mask |= sliding_attend_mask

            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)

            if attn_mask is not None:
                causal_mask = causal_mask.clone()
                if attn_mask.shape[-1] > target_len:
                    attn_mask = attn_mask[:, :target_len]
                mask_len = attn_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_len] + attn_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_len] = causal_mask[:, :, :, :mask_len].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask

class Qwen2VLForConditionalGeneration(Qwen2VLPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, generation_config):
        super().__init__(config, generation_config)
        self.visual = Qwen2VisionTransformerPretrainedModel(
            config.vision_config, generation_config
        )
        self.model = Qwen2VLModel(config, generation_config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.padding_side = "left"
        self.post_init()
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
    
    def get_output_embeddings(self):
        return self.lm_head
    def set_output_embeddings(self, new_embs):
        self.lm_head = new_embs

    def set_decoder(self, decoder):
        self.model = decoder
    def get_decoder(self):
        return self.model
    
    def get_rope_index( # RoPE Rotational Positional Encoding
        self, input_ids,
        image_grid_thw, video_grid_thw,
        attn_mask
    ):
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []

        if image_grid_thw is not None or video_grid_thw is not None:
            total_input_ids = input_ids

            position_ids = torch.ones(
                3, input_ids.shape[0], input_ids.shape[1], 
                dtype=input_ids.dtype, device=input_ids.device
            )
            
            img_idx, vid_idx = 0, 0

            for i, input_ids in enumerate(total_input_ids):
                if attn_mask is not None:
                    input_ids = input_ids[attn_mask[i] == 1]
                
                img_nums, vid_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                img_nums = (vision_tokens == image_token_id).sum()
                vid_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list = []
                st = 0
                remain_imgs, remain_vids = img_nums, vid_nums
                
                for _ in range(img_nums + vid_nums):
                    if image_token_id in input_tokens and remain_imgs > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1

                    if video_token_id in input_tokens and remain_vids > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1

                
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[img_idx][0],
                            image_grid_thw[img_idx][1],
                            image_grid_thw[img_idx][2],
                        )
                        vid_idx += 1
                        remain_vids -1
                        ed = ed_video
                    
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attn_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))

            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        
        else:
            if attn_mask is not None:
                position_ids = attn_mask.long().cumsum(-1) - 1
                position_ids.masked_fill(attn_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attn_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype
                )
            
            return position_ids, mrope_position_deltas
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        rope_deltas=None 
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_grid_thw = image_grid_thw.view(1, 3)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_type())
                video_embs = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embs = video_embs.to(inputs_embeds.device, inputs_embeds.dtype)                
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embs)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)
        
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states            
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states).float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fn = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fn(shift_logits, shift_labels)

        return Qwen2VLOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_kv=None,
        attn_mask=None,
        inputs_embs=None,
        cache_pos=None,
        pos_ids=None,
        use_cache=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs
    ):
        if past_kv is not None:
            if inputs_embs is not None:
                input_ids = input_ids[:, -cache_pos.shape[0]:]
            elif input_ids.shape[1] != cache_pos.shape[0]:
                input_ids = input_ids[:, cache_pos]
        
        rope_deltas = kwargs.get("rope_deltas", None)
        if attn_mask is not None and pos_ids is None:
            if cache_pos is None or (cache_pos is not None and cache_pos[0] == 0):
                pos_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attn_mask
                )
            else:
                batch_size, seq_len = input_ids.shape
                delta = (
                    cache_pos[0] + rope_deltas if cache_pos is not None and rope_deltas is not None else 0
                )
                pos_ids = torch.arange(seq_len, device=input_ids.device)
                pos_ids = pos_ids.view(1, -1).expand(batch_size, -1)
                pos_ids = pos_ids.add(delta)
                pos_ids = pos_ids.unsqueeze(0).expand(3, -1, -1)
        
        if cache_pos[0] != 0:
            pixel_values = None
            pixel_values_videos = None
        
        if inputs_embs is not None and cache_pos[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embs, "input_ids": None}
        else:
            model_inputs = {"inputs_embeds": None, "input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": pos_ids,
                "past_key_values": past_kv,
                "use_cache": use_cache,
                "attention_mask": attn_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "rope_deltas": rope_deltas,
            }
        )
        return model_inputs