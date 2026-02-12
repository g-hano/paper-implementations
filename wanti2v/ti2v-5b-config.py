class WanTI2V5BConfig:
    # t5
    tokenizer = 'google/umt5-xxl'

    # vae
    vae_stride = (4, 16, 16)

    # transformer settings
    patch_size = (1, 2, 2)
    dim = 3072
    ffn_dim = 14336
    freq_dim = 256
    num_heads = 24
    num_layers = 30
    window_size = (-1, -1)
    qk_norm = True
    cross_attn_norm = True
    eps = 1e-6

    # inference settings
    sample_fps = 24
    sample_shift = 5.0
    sample_steps = 50
    sample_guide_scale = 5.0
    frame_num = 121