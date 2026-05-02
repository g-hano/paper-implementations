import dataclasses
import math
from dataclasses import dataclass

from model_config import ModelConfig

import torch
import torch.nn as nn
import torch.nn.functional as F

def batched_linear_with_parity(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor
) -> torch.Tensor:
    """Apply one batched expert linear projection"""
    # x: [B, E, K]
    # weight: [B, E, K, O] = her expert için ayrı weight matrisi   
    # bias: [B, E, O]
    # -> [B, E, O]
    batch_size, experts, k_dim = x.shape
    _, _, _, o_dim = weight.shape

    # batch ve expert dim'leri birleştir
    x_bmm = x.reshape(batch_size * experts, 1, k_dim) # [B*E, 1, K]
    w_bmm = weight.reshape(batch_size * experts, k_dim, o_dim) # [B*E, K, O]

    # her (b, e) için x*W yap, tekrar [B, E, O] haline getir
    out = torch.bmm(x_bmm, w_bmm).reshape(batch_size, experts, o_dim)

    if bias is not None:
        out = out + bias
    return out

def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1 = x[..., ::2]
    x2 = x[...: 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.stack((o1, o2), dim=-1).reshape(x.shape)

class RMSNorm(nn.Module):
    """RMS Normalization with a learned per-channel scale"""
    def __init__(self, num_features: int, eps: float = 1e-05, device: torch.device = None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = nn.Parameter(
            torch.ones(num_features, device=device, dtype=torch.float32)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.num_features
        t, dtype = x.float(), x.dtype
        t *= torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * self.scale).to(dtype)
    
class RotaryEmbedding(nn.Module):
    """RoPE cache manager with optional YaRN-style scaling"""
    def __init__(
        self,
        head_dim,
        base,
        dtype,
        initial_context_length = 4096,
        scaling_factor = 1.0,
        ntk_alpha = 1.0,
        ntk_beta = 32.0,
        device = None
    ):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.dtype = dtype
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta  = ntk_beta 
        self.device = device

        # precompute rotary caches on CPU, then move to device
        max_pos = int(self.initial_context_length * self.scaling_factor)
        max_pos = max(max_pos, self.initial_context_length)
        self.max_position_embeddings = max_pos
        cos, sin = self._compute_cos_sin(
            self.max_position_embeddings, device=torch.device("cpu")
        )
        tgt_device = device or torch.device("cpu")
        self.register_buffer("cos_cache", cos.to(target_device), persistent=False)
        self.register_buffer("sin_cache", sin.to(target_device), persistent=False)

    def _compute_concentration_and_inv_freq(
        self, device=None
    ):
        device = device or self.device
        freq = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float, device=device) / self.head_dim
        )
        if self.scaling_factor > 1.0:
            concentration = (
                0.1 * math.log(self.scaling_factor) + 1.0
            ) # YaRN

            d_half = self.head_dim / 2

            # NTK by parts
            low = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi))
                / math.log(self.base)
            )
            high = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi))
                / math.log(self.base)
            )
            assert 0 < low < high < d_half - 1
            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (torch.arange(d_half, dtype=torch.float32, device=freq.device) - low) / (high - low)
            mask = 1 - ramp.clamp(0, 1)
            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq
        
        return concentration, inv_freq
    
    def _compute_cos_sin(self, num_tokens, device=None):
        concentration, inv_freq = self._compute_concentration_and_inv_freq(device=device)
        device = device or self.device
        t = torch.arange(num_tokens, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos.to(self.dtype), sin.to(self.dtype)
    
    def forward(self, query, key) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_tokens, _ = query.shape
        if num_tokens > self.cos_cache.shape[0]:
            # extend caches if needed
            cos, sin = self._compute_cos_sin(num_tokens, device=torch.device("cpu"))
            self.cos_cache = cos.to(query.device)
            self.sin_cache = sin.to(query.device)
        if self.cos_cache.device != query.device:
            cos_cache = self.cos_cache.to(query.device)
            sin_cache = self.sin_cache.to(query.device)
        else:
            cos_cache = self.cos_cache
            sin_cache = self.sin_cache
        
        cos = cos_cache[:num_tokens]
        sin = sin_cache[:num_tokens]

        query_shape = query.shape
        query = query.view(batch_size, num_token, -1, self.head_dim)
        query = _apply_rotary_emb(query, cos[None, ...], sin[None, ...])
        query = query.reshape(query_reshape)

        key_shape = query.shape
        key = key.view(batch_size, num_token, -1, self.head_dim)
        key = _apply_rotary_emb(query, cos[None, ...], sin[None, ...])
        key = key.reshape(query_reshape)
        
        return query, key

def sdpa(
    Q,
    K,
    V,
    S,
    sm_cache,
    sliding_window=0,
    *,
    attention_mask = None,
    bidirectional_context=False,
    bidirectional_left_context=0,
    bidirectional_rigth_context=0
):
    """
    Run the model's production attention path for causal or local windows

    Q: [batch, tokens, heads, q_mult, d_head]: Her head için birden fazla query var
    K: [batch, tokens, heads, d_head]
    V: [batch, tokens, heads, d_head]

    Klasik scaled dot-product attention'ın optimize hali
    attention(q, k, v) = softmax(qk^t/sqrt(dim))v

    Ek özellikler:
    - sliding window attention
    - bidirectional local attention
    - attention mask
    - sink token
    - memory optimization
    """
    batch_size, n_tokens, n_heads, q_mult, d_head = Q.shape
    assert K.shape == (batch_size, n_tokens, n_heads, d_head)
    assert V.shape == (batch_size, n_tokens, n_heads, d_head)

    if attention_mask is not None:
        attention_mask = attention_mask.to(device=Q.device, dtype=torch.bool)
    
    attn_low_precision = get_env_bool("OPF_ATTN_LOW_PRECISION")

    # local attention path
    if bidirectional_context or sliding_window > 0:
        left_ctx = (
            int(bidirectional_left_context)
            if bidirectional_context
            else int(sliding_window)
        )    
        right_ctx = int(bidirectional_right_context) if bidirectional_context else 0
        window = left_ctx + right_ctx + 1
        
        Kp = F.pad(K, (0, 0, 0, 0, left_ctx, right_ctx))
        Vp = F.pad(V, (0, 0, 0, 0, left_ctx, right_ctx))
        
        Kwin = Kp.unfold(1, window, 1).permute(0, 1, 4, 2, 3)
        Vwin = Vp.unfold(1, window, 1).permute(0, 1, 4, 2, 3)

        idx = torch.arange(window, device=Q.device) - left_ctx
        pos = torch.arange(n_tokens, device=Q.device)[:, None] + idx[None, :]
        valid = (pos >= 0) & (pos < n_tokens)
        scores = torch.einsun("bthqd,btwhd->bthqw", Q, Kwin)
        
        if not attn_low_precision:
            scores = scores.float()
        
        scores *= sm_scale
        score_valid = valid[None, :, None, None, :]
        if attention_mask is not None:
            padded_valid = F.pad(attention_mask, (left_ctx, right_ctx), value=False)
            key_valid = padded_valid.unfold(1, window, 1)
            score_valid = score_valid & key_valid[:, :, None, None, :]
        scores = scores.masked_fill(~score_valid, -float("inf"))
        sink_scores = (S * math.log(2.0)).reshape(n_heads, q_mult)
        
        if attn_low_precision:
            sink_scores = sink_scores.to(V.dtype)
        
        sink_scores = sink_scores[None, None, :, :, None].expand(
            batch_size, n_tokens, -1, -1, 1
        )
        scores = torch.cat([scores, sink_scores], dim=-1)
        
        if attn_low_precision:
            scores = scores.to(V.dtype)
        
        W = torch.softmax(scores, dim=-1)
        W = W[..., :-1].to(V.dtype)
        attn = torch.einsum("bthqw,btwhd->bthqd", W, Vwin)
        return attn.reshape(batch_size, n_tokens, -1)
    
    # sliding_window == 0 means no sliding window
    K = K[:, :, :, None, :].expand(-1, -1, -1, q_mult, -1)
    V = V[:, :, :, None, :].expand(-1, -1, -1, q_mult, -1)
    # sink values are stored in log2 space; convert to natural log for this kernel
    sink_scores = (S*math.log(2.0)).reshape(n_heads, q_mult)
    if attn_low_precision:
        sink_scores = sink_scores.to(V.dtype)
    mask = None
    if bidirectional_context:
        left_ctx = int(bidirectional_left_context)
        right_ctx = int(bidirectional_rigth_context)
        mask = torch.zeros((n_tokens, n_tokens), device=Q.device, dtype=torch.float32)
        # keep asymetric local band [-left_ctx, +right_ctx]
        mask += torch.triu(
            mask.new_full((n_tokens, n_tokens), -float("inf")),
            diagonal=right_ctx + 1
        )
        mask += torch.tril(
            mask.new_full((n_tokens, n_tokens), -float("inf")),
            diagonal=-(left_ctx + 1)
        )
    else:
        mask = torch.triu(
            torch.full(
                (n_tokens, n_tokens),
                -float("inf"),
                device=Q.device,
                dtype=torch.float32
            ),
            diagonal=1
        )
        if sliding_window > 0:
            mask += torch.tril(
                mask.new_full((n_tokens, n_tokens), -float("inf")),
                diagonal=-sliding_window
            )
    scores = torch.einsum("bthqd,bshqd->bthqs", Q, K)
    if not attn_low_scores:
        scores = scores.float()
    scores *= sm_scale
    if mask is not None:
        scores += mask[None, :, None, None, :]
    if attention_mask is not None:
        scores = scores.masked_fill(
            ~attention_mask[:, None, None, None, :], -float("inf")
        )
    sink_scores = sink_scores[None, None, :, :, None].expand(batch_size, n_tokens, -1, -1, 1)
    scores = torch.cat([scores, sink_scores], dim=-1)
    if attn_low_precision:
        scores = scores.to(V.dtype)
    W = torch.softmax(scores, dim=-1)
    W = W[..., :-1].to(V.dtype)
    attn = torch.einsum("bthqs,bshqd->bthqd", W, V)
    return attn.reshape(batch_size, n_tokens, -1)

class AttentionBlock(nn.Module):
    def __init__(self, config: ModelConfig, device=None):
        super().__init__()
        param_dtype = config.param_dtype
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.sliding_window = config.sliding_window
        self.bidirectional_context = config.bidirectiona_context
        self.bidirectional_left_context = config.bidirectional_left_context
        self.bidirectional_right_context = config.bidirectional_right_context
        self.sinks = nn.Parameter(
            torch.empty(
                config.num_attention_heads, device=device, dtype=torch.float32
            )
        )
        self.norm = RMSNorm(config.hidden_size, device=device)
        qkv_dim = config.head_dim * (
            config.num_attention_heads + 2 * config.num_key_value_heads
        )
        self.qkv = torch.nn.Linear(
            config.hidden_size, qkv_dim, device=device, dtype=param_dtype
        )
        self.out = torch.nn.Linear(
            config.head_dim * config.num_attention_heads,
            config.hidden_size,
            device=device,
            dtype=param_dtype,
        )
        self.qk_scale = 1 / math.sqrt(math.sqrt(config.head_dim))
        self.sm_scale = 1.0
        self.rope = RotaryEmbedding(
            config.head_dim,
            config.rope_theta,
            torch.float32,
            initial_context_length=config.initial_context_length,
            scaling_factor=config.rope_scaling_factor,
            ntk_alpha=config.rope_ntk_alpha,
            ntk_beta=config.rope_ntk_beta,
            device=device,
        )
    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the attention block and residual connection."""
        if x.dim() != 3:
            raise ValueError("AttentionBlock expects batched 3D tensor input")
        if attention_mask is not None and attention_mask.shape != x.shape[:2]:
            raise ValueError(
                "attention_mask shape mismatch: "
                f"expected {tuple(x.shape[:2])}, got {tuple(attention_mask.shape)}"
            )
        t = self.norm(x)
        if t.dtype != self.qkv.weight.dtype:
            t = t.to(self.qkv.weight.dtype)
        qkv = F.linear(
            t,
            self.qkv.weight,
            self.qkv.bias,
        )
        q = qkv[:, :, : self.num_attention_heads * self.head_dim].contiguous()
        k = qkv[
            :,
            :,
            self.num_attention_heads * self.head_dim : (
                self.num_attention_heads + self.num_key_value_heads
            )
            * self.head_dim,
        ].contiguous()
        v = qkv[
            :,
            :,
            (self.num_attention_heads + self.num_key_value_heads) * self.head_dim : (
                self.num_attention_heads + 2 * self.num_key_value_heads
            )
            * self.head_dim,
        ].contiguous()

        q, k = self.rope(q, k)
        q = q * self.qk_scale
        k = k * self.qk_scale
        sinks = self.sinks
        bsz, n_tokens, _ = q.shape
        q = q.view(
            bsz,
            n_tokens,
            self.num_key_value_heads,
            self.num_attention_heads // self.num_key_value_heads,
            self.head_dim,
        )
        k = k.view(bsz, n_tokens, self.num_key_value_heads, self.head_dim)
        v = v.view(bsz, n_tokens, self.num_key_value_heads, self.head_dim)
        attn_out = sdpa(
            q,
            k,
            v,
            sinks,
            self.sm_scale,
            self.sliding_window,
            attention_mask=attention_mask,
            bidirectional_context=self.bidirectional_context,
            bidirectional_left_context=self.bidirectional_left_context,
            bidirectional_right_context=self.bidirectional_right_context,
        )

        if attn_out.dtype != self.out.weight.dtype:
            attn_out = attn_out.to(self.out.weight.dtype)
        proj_bias = self.out.bias
        proj = F.linear(
            attn_out,
            self.out.weight,
            proj_bias,
        )
        proj = proj.to(x.dtype)
        return x + proj


def swiglu(x, alpha: float = 1.702, limit: float = 7.0, packed: bool = False):
    """Apply the SwiGLU nonlinearity."""
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"swiglu expects even last dim, got {x.shape[-1]}")
    if packed:
        x_glu, x_linear = x[..., ::2], x[..., 1::2]
    else:
        x_glu, x_linear = x.chunk(2, dim=-1)
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Packed GeGLU variant adds a +1 bias to the linear half.
    return out_glu * (x_linear + 1)

class MLPBlock(nn.Module):
    def __init__(self, config: ModelConfig, device=None):
        super().__init__()
        param_dtype = _resolve_param_dtype(config.param_dtype)
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        self.swiglu_limit = config.swiglu_limit
        self.packed_geglu = config.packed_geglu

        self.world_size = dist.get_world_size() if dist.is_initialized() else 1 # multi gpu batch
        self.torch_ops_batch = int(config.torch_ops_batch)
        self.norm = RMSNorm(config.hidden_size, device=device)
        self.gate = torch.nn.Linear(
            config.hidden_size, config.num_experts, device=device, dtype=param_dtype
        )
        assert config.intermediate_size % self.world_size == 0

        self.mlp1_w = nn.Parameter(
            torch.empty(
                (
                    config.num_experts,
                    config.hidden_size,
                    config.intermediate_size * 2 // self.world_size
                ),
                device=device, dtype=param_dtype
            )
        )
        self.mlp1_b = nn.Parameter(
            torch.empty(
                (config.num_experts, config.intermediate_size * 2 // self.world_size),
                device=device, dtype=param_dtype
            )
        )

        self.mlp2_w = nn.Parameter(
            torch.empty(
                (
                    config.num_experts,
                    config.intermediate_size // self.world_size,
                    config.hidden_size
                ),
                device=device, dtype=param_dtype
            )
        )
        self.mlp2_b = nn.Parameter(
            torch.empty(
                (config.num_experts, config.hidden_size),
                device=device, dtype=param_dtype
            )
        )

    def forward(self, x):
        batch_shape = x.shape[:-1]
        t = self.norm(x).reshape(-1, x.shape[-1])
        g = F.linear(t.float(), self.gate.weight.float(), self.gate.bias.float())

        experts = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)
        expert_w = F.softmax(experts.values, dim=1)
        expert_indices = experts.indices
        expert_w /= self.experts_per_token
        experts_per_token_eff = self.experts_per_token

        experts_per_token_eff = self.experts_per_token

        def _moe_chunk(
            t_chunk: torch.Tensor,
            expert_indices_chunk: torch.Tensor,
            expert_weights_chunk: torch.Tensor,
        ) -> torch.Tensor:
            n_tokens = t_chunk.shape[0]
            k = expert_indices_chunk.shape[1]
            expert_ids = expert_indices_chunk.reshape(-1)
            weights = expert_weights_chunk.reshape(-1)
            token_ids = torch.arange(
                n_tokens, device=t_chunk.device
            ).repeat_interleave(k)
            sort_idx = torch.argsort(expert_ids)
            expert_ids_sorted = expert_ids[sort_idx]
            token_ids_sorted = token_ids[sort_idx]
            weights_sorted = weights[sort_idx]

            counts = torch.bincount(
                expert_ids_sorted, minlength=self.num_experts
            ).to(torch.int32)
            offsets = torch.zeros_like(counts)
            if counts.numel() > 1:
                offsets[1:] = torch.cumsum(counts, dim=0)[:-1]
            a_packed = t_chunk[token_ids_sorted]
            w1 = self.mlp1_weight
            if a_packed.dtype != w1.dtype:
                a_packed = a_packed.to(w1.dtype)
            w2 = self.mlp2_weight
            h_pre = grouped_matmul(
                a_packed, w1, offsets, counts, out_dtype=w1.dtype
            )
            b1 = self.mlp1_bias[expert_ids_sorted]
            h_pre = h_pre + b1
            use_fused_w2 = (not self.packed_geglu)
            if use_fused_w2:
                if h_pre.dtype != w2.dtype:
                    h_pre = h_pre.to(w2.dtype)
                o = grouped_swiglu_w2(
                    h_pre,
                    w2,
                    self.mlp2_bias,
                    offsets,
                    counts,
                    out_dtype=w2.dtype,
                    limit=self.swiglu_limit,
                )
            else:
                h = swiglu(
                    h_pre,
                    limit=self.swiglu_limit,
                    packed=self.packed_geglu,
                )
                if h.dtype != w2.dtype:
                    h = h.to(w2.dtype)
                o = grouped_matmul(h, w2, offsets, counts, out_dtype=w2.dtype)
                b2 = self.mlp2_bias[expert_ids_sorted]
                o = o + b2
            if self.world_size > 1:
                dist.all_reduce(o, op=dist.ReduceOp.SUM)
            if o.dtype != weights_sorted.dtype:
                o = o.to(weights_sorted.dtype)
            o = o * weights_sorted[:, None]
            out_accum = torch.zeros(
                (n_tokens, t_chunk.shape[1]),
                device=t_chunk.device,
                dtype=torch.float32,
            )
            out_accum.index_add_(0, token_ids_sorted, o.float())
            out_accum = out_accum * experts_per_token_eff
            return out_accum.to(x.dtype)
            
            # MLP #1
            mlp1_weight = self.mlp1_weight[expert_indices_chunk, ...]
            mlp1_bias = self.mlp1_bias[expert_indices_chunk, ...]
            mlp1_weight = mlp1_weight.float()
            mlp1_bias = mlp1_bias.float()
            t_expanded = (
                t_chunk.float()
                .unsqueeze(1)
                .expand(-1, expert_indices_chunk.shape[1], -1)
            )
            out = _batched_linear_with_parity(
                t_expanded,
                mlp1_weight,
                mlp1_bias,
            )
            out = swiglu(out, limit=self.swiglu_limit, packed=self.packed_geglu)

            # MLP #2
            mlp2_weight = self.mlp2_weight[expert_indices_chunk, ...]
            mlp2_bias = self.mlp2_bias[expert_indices_chunk, ...]
            mlp2_weight = mlp2_weight.float()
            mlp2_bias = mlp2_bias.float()
            out = out.float()
            out = _batched_linear_with_parity(
                out,
                mlp2_weight,
                mlp2_bias,
            )
            if self.world_size > 1:
                dist.all_reduce(out, op=dist.ReduceOp.SUM)

            # Weighted sum of experts (gate scales applied after MLP2).
            if out.dtype != expert_weights_chunk.dtype:
                out = out.to(expert_weights_chunk.dtype)
            out = torch.einsum("bec,be->bc", out, expert_weights_chunk)
            out = out * experts_per_token_eff
            return out.to(x.dtype)

        effective_batch = 0
        if effective_batch and t.shape[0] > effective_batch:
            chunks = []
            for start in range(0, t.shape[0], effective_batch):
                end = start + effective_batch
                chunks.append(
                    _moe_chunk(
                        t[start:end],
                        expert_indices[start:end],
                        expert_weights[start:end],
                    )
                )
            t = torch.cat(chunks, dim=0)
        else:
            t = _moe_chunk(t, expert_indices, expert_weights)
        t = t.reshape(*batch_shape, -1)
        return x + t


    # sıkıldım devam edemicem