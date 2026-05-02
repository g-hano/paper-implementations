from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for an OPF transformer checkpoint"""

    model_type = "privacy_filter"
    num_hidden_layers = 36
    num_experts = 128
    experts_per_token = 4
    vocab_size = 201_088
    num_labels = None
    hidden_size = 2880
    intermediate_size = 2800
    swiglu_limit = 7.0
    packed_geglu = False
    head_dim = 64
    num_attention_heads = 64
    num_key_value_heads = 8
    sliding_window = 128
    bidirectional_context = False
    bidirectional_left_context = 0
    bidirectional_right_context = 0
    initial_context_length = 4096
    rope_theta = 150000.0
    rope_scaling_factor = 32.0
    rope_ntk_alpha = 1.0
    rope_ntk_beta = 32.0
    torch_ops_batch = 32
    param_dtype = "bfloat16"