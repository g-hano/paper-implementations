# Bidirectional Neural Machine Translation with Enhanced Transformer

This repository contains an implementation of the Transformer architecture from ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) with several modern enhancements for bidirectional translation between French and Russian.

## Key Features

- **Bidirectional Translation**: Single model capable of translating between French↔Russian in both directions
- **Enhanced Architecture**: Several improvements over the original Transformer:
  - Replaced sinusoidal positional embeddings with Rotary Position Embeddings (RoPE) from ["RoFormer: Enhanced Transformer with Rotary Position Embedding"](https://arxiv.org/abs/2104.09864)
  - Implemented Sliding Window Attention mechanism from ["Longformer: The Long-Document Transformer"](https://arxiv.org/pdf/2004.05150)
  - Pre-normalization architecture for improved training stability
  - Direction-specific embeddings to handle bidirectional translation

## Model Architecture

The model follows the Transformer architecture with the following modifications:
- Encoder and Decoder with sliding window self-attention
- RoPE for better position encoding
- Language direction tokens (``[FR2RU]`` and ``[RU2FR]``) to control translation direction
- Layer normalization before attention and feed-forward layers
- Shared vocabulary between languages

## Training

The model was trained on the OPUS Books dataset using:
- AdamW optimizer with cosine learning rate scheduling
- Gradient clipping for stability
- Mixed training strategy (both translation directions in each batch)

### Training Progress
<img src="https://github.com/g-hano/paper-implementations/blob/main/machine-translation/plot/loss_plot.png" width="800"/>

## Project Structure

```
machine-translation/
├── tokenizers/            # Pre-trained tokenizers for French and Russian
├── transformers_model.py  # Core model architecture
├── transformers_train.py  # Training loop and utilities
├── transformers_dataset.py # Dataset preprocessing and loading
├── transformers_inference.py # Inference and translation utilities
└── config.yaml           # Model and training configuration
```

## Usage

### Requirements
```bash
pip install torch datasets tqdm spacy PyYAML
python -m spacy download fr_core_news_sm
python -m spacy download ru_core_news_sm
```

### Training
```bash
python transformers_train.py --config config.yaml
```

### Inference
```python
from transformers_inference import load_model_for_inference, translate_with_beam_search
from transformers_dataset import get_tokenizers
from datasets import load_dataset

# Load model and tokenizers
model, config = load_model_for_inference('path_to_checkpoint.pt', device)
dataset = load_dataset("opus_books", "fr-ru", split="train")
fr_tokenizer, ru_tokenizer = get_tokenizers(dataset)

# Translate French to Russian
fr_text = "Bonjour le monde"
ru_translation = translate_with_beam_search(
    model, fr_text, fr_tokenizer, ru_tokenizer,
    seq_len=128, device=device, direction="fr2ru"
)

# Translate Russian to French
ru_text = "Привет мир"
fr_translation = translate_with_beam_search(
    model, ru_text, ru_tokenizer, fr_tokenizer,
    seq_len=128, device=device, direction="ru2fr"
)
```
were trained on the OPUS Books dataset and handle special tokens for both languages.

## Contributing

Feel free to open issues or submit pull requests for improvements or bug fixes.

## License

[MIT License](LICENSE)
