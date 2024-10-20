from typing import Union

class Qwen2VLVisionConfig:
    def __init__(
        self, depth=32, embed_dim=1280,
        hidden_size=3584, hidden_act="quick_gelu",
        mlp_ratio=4, num_heads=16, in_channels=3,
        patch_size=14, spatial_merge_size=2, temporal_patch_size=2,
        **kwargs
    ):
        self.depth = depth
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        

class Qwen2VLConfig:
    keys_to_ignore_at_inference = ["past_key_values"]
    def __init__(self,
                 vocab_size=152064,
                 hidden_size=8192,
                 intermediate_size=29568,
                 num_hidden_layers=80,
                 num_attention_heads=64,
                 num_key_value_heads=8,
                 hidden_act="silu",
                 max_position_embeddings=32768,
                 initializer_range=0.02,
                 rms_norm_eps=1e-05,
                 use_cache=True,
                 tie_word_embeddings=False,
                 rope_theta=1000000.0,
                 use_sliding_window=False,
                 sliding_window=4096,
                 max_window_layers=80,
                 attention_dropout=0.0,
                 vision_config=None,
                 rope_scaling=None,
                 ):
        self.vision_config = Qwen2VLVisionConfig(**vision_config)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads if num_key_value_heads is None else num_key_value_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.rope_scaling = rope_scaling
        self.attn_implementation = "eager"