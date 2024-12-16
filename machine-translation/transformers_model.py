import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class InputEmbeddings(nn.Module):
    def __init__(self, dim_model: int, vocab_size: int) -> None:
        super().__init__()
        self.dim_model = dim_model
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, dim_model)

    def forward(self, hidden_states: torch.Tensor):
        return self.embed(hidden_states) * math.sqrt(self.dim_model)
    
class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        position_indices = torch.arange(0, d_model, 2).float()
        freq_indices = position_indices / d_model
        inv_freq = 1.0 / (10000 ** freq_indices)
        
        self.register_buffer("inv_freq", inv_freq)

    def _compute_rope(self, positions):
        freqs = torch.einsum('i,j->ij', positions.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return torch.cos(emb), torch.sin(emb)
    
    def _rotate_half(self, x):
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, x: torch.Tensor, start_pos: int = 0):
        seq_len = x.shape[1]
        positions = torch.arange(start_pos, start_pos + seq_len, device=x.device)
        
        cos, sin = self._compute_rope(positions)
        cos = cos.unsqueeze(0)  # [1, seq_len, d_model]
        sin = sin.unsqueeze(0)  # [1, seq_len, d_model]
        
        return x * cos + self._rotate_half(x) * sin
    
class MultiWindowHeadCrossAttention(nn.Module):
    def __init__(self, dim_model: int, num_heads: int, window_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.dim_model = dim_model
        self.num_heads = num_heads       
        assert dim_model % num_heads == 0, "dim_model must be divisible by num_heads"
        self.head_dim = dim_model // num_heads
        self.window_size = window_size

        self.q_proj = nn.Linear(dim_model, dim_model)
        self.k_proj = nn.Linear(dim_model, dim_model) 
        self.v_proj = nn.Linear(dim_model, dim_model)
        self.o_proj = nn.Linear(dim_model, dim_model)

        self.dropout = nn.Dropout(dropout)
        
    def split_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim_model = hidden_states.size()
        hidden_states = hidden_states.view(batch, seq_len, self.num_heads, self.head_dim)
        hidden_states = hidden_states.transpose(1, 2)
        # [batch, self.num_heads, seq_len, self.head_dim]
        return hidden_states

    def combine_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, num_heads, seq_len, head_dim = hidden_states.size()
        hidden_states = hidden_states.transpose(1, 2).contiguous()
        hidden_states = hidden_states.view(batch, seq_len, self.dim_model)
        return hidden_states
    
    def scaled_dot_product_attention_window(self, Q, K, V, is_causal=False):
        batch_size, num_heads, seq_len, head_dim = Q.size()

        assert self.window_size > 0, f"Window size must be positive, got {self.window_size}"
        assert seq_len > 0, f"Sequence length must be positive, got {seq_len}"
        
        # Pad K and V at the beginning for lookback window
        K_padded = F.pad(K, (0, 0, self.window_size-1, 0))  # pad sequence dimension
        V_padded = F.pad(V, (0, 0, self.window_size-1, 0))

        outputs = []
        for i in range(seq_len):
            # Define window boundaries
            window_end = i + self.window_size
            assert window_end <= K_padded.size(2), f"Window end {window_end} exceeds padded sequence length {K_padded.size(2)}"

            # Extract current window
            q = Q[:, :, i:i+1, :]  # Shape: [batch_size, num_heads, 1, head_dim]
            k = K_padded[:, :, i:i+window_end, :]  # Shape: [batch_size, num_heads, window_size, head_dim]
            v = V_padded[:, :, i:i+window_end, :]  # Shape: [batch_size, num_heads, window_size, head_dim]

            # Calculate attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # Apply causal masking if needed
            if is_causal:
                mask = torch.zeros(1, 1, 1, scores.shape[-1], device=q.device)
                mask[:, :, :, :min(i + 1, self.window_size)] = 1
                scores = scores.masked_fill(mask == 0, float('-inf'))
                
            # Calculate attention weights and apply dropout
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Calculate output for current position
            output = torch.matmul(attn_weights, v)
            outputs.append(output)
        
        # Concatenate all outputs
        outputs = torch.cat(outputs, dim=2)
        return outputs

    def forward(self, query, kv, is_causal=False):
        Q = self.split_heads(self.q_proj(query))
        K = self.split_heads(self.k_proj(kv))
        V = self.split_heads(self.v_proj(kv))
        attn_output = self.scaled_dot_product_attention_window(
            Q, K, V, is_causal=is_causal
        )
        output = self.combine_heads(attn_output)
        output = self.o_proj(output)
        return output
    
class FeedForward(nn.Module):
    def __init__(self, dim_model: int, ff_dim: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(dim_model, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, dim_model)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear1(hidden_states)
        hidden_states = torch.relu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.linear2(hidden_states)
        return hidden_states
    
class EncoderBlock(nn.Module):
    def __init__(self, dim_model: int, num_heads: int, window_size: int, dropout: float):
        super().__init__()
        self.dim_model = dim_model
        self.norm1 = nn.LayerNorm(dim_model)
        # [batch_size, seq_len, dim_model]
        self.mhca = MultiWindowHeadCrossAttention(dim_model, num_heads, window_size, dropout)

        self.norm2 = nn.LayerNorm(dim_model)
        self.feed_forward = FeedForward(dim_model, dim_model*4, dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.mhca(hidden_states, hidden_states) + residual

        residual = hidden_states

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.feed_forward(hidden_states) + residual

        return hidden_states
    
class Encoder(nn.Module):
    def __init__(self, num_layers: int, dim_model: int, num_heads: int, window_size: int, dropout: float):
        super().__init__()
        self.dim_model = dim_model
        self.layers = nn.ModuleList(
            [
                EncoderBlock(dim_model, num_heads, window_size, dropout) 
                for _ in range(num_layers)
            ]
        )
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states
                )
            else:
                hidden_states = layer(hidden_states)
        return hidden_states
    
class DecoderBlock(nn.Module):
    def __init__(self, dim_model: int, num_heads: int, window_size: int, dropout: float):
        super().__init__()
        self.dim_model = dim_model
        self.norm1 = nn.LayerNorm(dim_model)
        self.mhca1 = MultiWindowHeadCrossAttention(dim_model, num_heads, window_size, dropout)

        self.norm2 = nn.LayerNorm(dim_model)
        self.mhca2 = MultiWindowHeadCrossAttention(dim_model, num_heads, window_size, dropout)

        self.norm3 = nn.LayerNorm(dim_model)
        self.feed_forward = FeedForward(dim_model, dim_model*4, dropout)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        residual = target

        target = self.norm1(target)
        target = self.mhca1(target, target, is_causal=True) + residual

        residual = target

        target = self.norm2(target)
        target = self.mhca2(target, source) + residual

        residual = target

        target = self.norm3(target)
        target = self.feed_forward(target) + residual

        return target
    
class Decoder(nn.Module):
    def __init__(self, num_layers: int, dim_model: int, num_heads: int, window_size: int, dropout: float):
        super().__init__()
        self.dim_model = dim_model
        self.layers = nn.ModuleList(
            [
                DecoderBlock(dim_model, num_heads, window_size, dropout)
                for _ in range(num_layers)
            ]
        )
        self.gradient_checkpointing = False

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # Use checkpoint to save memory during training
                target = torch.utils.checkpoint.checkpoint(
                    # We need to wrap the layer call in a lambda because
                    # checkpoint doesn't support multiple input arguments directly
                    lambda s, t: layer(s, t),
                    source,
                    target
                )
            else:
                target = layer(source, target)
        return target
    
class Transformer(nn.Module):
    def __init__(
        self, 
        encoder: Encoder, decoder: Decoder, 
        source_embed: InputEmbeddings, target_embed: InputEmbeddings,
        source_pos: RotaryPositionalEncoding, target_pos: RotaryPositionalEncoding,
        vocab_size: int
    ) -> torch.Tensor:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        self.source_embed = source_embed
        self.source_pos = source_pos

        self.target_embed = target_embed
        self.target_pos = target_pos

        # Add language embeddings
        self.language_embedding = nn.Embedding(6, encoder.layers[0].dim_model)  # 6 special tokens
        # Initialize direction token IDs
        self.fr2ru_token_id = 4  # [FR2RU] token ID
        self.ru2fr_token_id = 5  # [RU2FR] token ID
        self.output_proj = nn.Linear(decoder.layers[0].dim_model, vocab_size)

    def encode(self, source):
        #print(f"{source.shape=}")
        direction_tokens = source[:, 1].long()
        #print(f"{direction_tokens.shape=}")
        
        lang_embeddings = self.language_embedding(direction_tokens)
        lang_embeddings = lang_embeddings.unsqueeze(1)
        #print(f"{lang_embeddings.shape=}")
        
        token_embeddings = self.source_embed(source)
        #print(f"{token_embeddings.shape=}")
        
        lang_embeddings = lang_embeddings.expand(
            -1,  
            token_embeddings.size(1), 
            -1   
        )
        #print(f"{lang_embeddings.shape=}")

        combined_embeddings = token_embeddings + lang_embeddings
        combined_embeddings = self.source_pos(combined_embeddings)
        #print(f"{combined_embeddings.shape=}")
        to_return = self.encoder(combined_embeddings)
        #print(f"{to_return.shape=}")
        return to_return, direction_tokens

    def decode(self, encoder_output, target, direction_tokens):  # Added direction_tokens parameter
        # Use the direction tokens passed from encode
        # No need to extract from encoder output anymore
        
        # Get language embeddings
        lang_embeddings = self.language_embedding(direction_tokens)  # [batch_size, dim_model]
        lang_embeddings = lang_embeddings.unsqueeze(1)  # [batch_size, 1, dim_model]
        
        # Get token embeddings
        token_embeddings = self.target_embed(target)  # [batch_size, seq_len, dim_model]
        
        # Expand language embeddings
        lang_embeddings = lang_embeddings.expand(
            -1,  # keep batch size
            token_embeddings.size(1),  # expand to match sequence length
            -1   # keep embedding dimension
        )
        
        # Combine embeddings
        combined_embeddings = token_embeddings + lang_embeddings
        combined_embeddings = self.target_pos(combined_embeddings)
        
        decoder_output = self.decoder(encoder_output, combined_embeddings)
        return self.output_proj(decoder_output)

    
    def forward(
        self, 
        source: torch.Tensor, 
        target: torch.Tensor=None
    ) -> torch.Tensor:
        encoder_output, direction_tokens = self.encode(source)
        if target is not None:
            return self.decode(encoder_output, target, direction_tokens)
        else:
            batch_size = source.shape[0]
            current_sequence = torch.full(
                (batch_size, 1), 
                fill_value=2,  # SOS token
                device=source.device
            )

            # Use greedy decoding during inference
            max_length = source.shape[1] + 50
            for _ in range(max_length):
                outputs = self.decode(encoder_output, current_sequence, direction_tokens)
                next_token = outputs[:, -1, :].argmax(dim=-1, keepdim=True)
                current_sequence = torch.cat([current_sequence, next_token], dim=1)
                
                if (next_token == 3).all():
                    break
                    
            return current_sequence
        

DIM_MODEL = 512
NUM_HEADS = 8
WINDOW_SIZE = 3
DROPOUT = 0.1
FF_DIM = 2048
SOURCE_VOCAB_SIZE = 10_000
SOURCE_SEQ_LEN = 1024
TARGET_VOCAB_SIZE = 10_000
TARGET_SEQ_LEN = 1024
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6

encoder = Encoder(NUM_ENCODER_LAYERS, DIM_MODEL, NUM_HEADS, WINDOW_SIZE, DROPOUT)
decoder = Decoder(NUM_DECODER_LAYERS, DIM_MODEL, NUM_HEADS, WINDOW_SIZE, DROPOUT)
source_embeddings = InputEmbeddings(DIM_MODEL, SOURCE_VOCAB_SIZE)
target_embeddings = InputEmbeddings(DIM_MODEL, TARGET_VOCAB_SIZE)
source_positions = RotaryPositionalEncoding(DIM_MODEL, SOURCE_SEQ_LEN)
target_positions = RotaryPositionalEncoding(DIM_MODEL, TARGET_SEQ_LEN)
transformer = Transformer(
    encoder, decoder, 
    source_embeddings, target_embeddings, 
    source_positions, target_positions, 
    TARGET_VOCAB_SIZE).to('cuda')