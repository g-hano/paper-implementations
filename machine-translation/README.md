I was implementing `Attention is All You Need` paper and I tried to improve it.
The changes I made to original Transformers:
- Changed sinusodal positional embeddings to RoPE ([`RoFormer: Enhanced Transformer with Rotary Position Embedding`](https://arxiv.org/abs/2104.09864))
- Changed attention mechanism to Sliding Window Attention ([`Longformer: The Long-Document Transformer`](https://arxiv.org/pdf/2004.05150))
- Normalization before attention
- I wanted my model to be able to translate bidirectional, in my case its French to Russian and Russian to French
