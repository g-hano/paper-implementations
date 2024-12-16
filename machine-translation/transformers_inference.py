from transformers_model import (
    Transformer, Encoder, Decoder, 
    InputEmbeddings, RotaryPositionalEncoding
)
from transformers_dataset import create_tokenizers
import torch
from datasets import load_dataset

def greedy_decode(model, source_text, tokenizer_src, tokenizer_tgt, max_len, device, mode="fr2ru"):
    """
    Performs greedy decoding for translation.
    
    Args:
        model: The transformer model
        source_text: Input text to translate
        tokenizer_src: Source language tokenizer
        tokenizer_tgt: Target language tokenizer
        max_len: Maximum length of generated sequence
        device: Device to run the model on
    """
    model.eval()
    
    # Tokenize source text
    source_tokens = tokenizer_src.encode(source_text).ids
    
    # Add SOS, direction token, and EOS
    if mode == "fr2ru":  # French to Russian
        direction_token = tokenizer_src.token_to_id("[FR2RU]")
    else:  # Russian to French
        direction_token = tokenizer_src.token_to_id("[RU2FR]")
        
    source = torch.tensor([[
        tokenizer_src.token_to_id("[SOS]"),
        direction_token,
        *source_tokens,
        tokenizer_src.token_to_id("[EOS]")
    ]], dtype=torch.long).to(device)
    
    # Encode the source sequence
    encoder_output = model.encode(source)
    
    # Initialize decoder input with SOS token
    decoder_input = torch.tensor([[tokenizer_tgt.token_to_id("[SOS]")]], 
                               dtype=torch.long).to(device)
    
    output_tokens = []
    
    with torch.no_grad():
        for _ in range(max_len):
            # Get model predictions
            output = model.decode(encoder_output, decoder_input)
            
            # Get the next token prediction
            next_token_logits = output[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            # Break if EOS token is generated
            if next_token.item() == tokenizer_tgt.token_to_id("[EOS]"):
                break
                
            # Add predicted token to decoder input for next iteration
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
            output_tokens.append(next_token.item())
    
    # Convert token IDs back to text
    translated_tokens = [tokenizer_tgt.id_to_token(token_id) for token_id in output_tokens]
    translated_text = ' '.join(translated_tokens)
    
    return translated_text

def translate(model, text, src_tokenizer, tgt_tokenizer, seq_len, device, direction="fr2ru"):
    model.eval()
    
    # Prepare special tokens
    sos_token = torch.tensor([src_tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
    eos_token = torch.tensor([src_tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
    pad_token = torch.tensor([src_tokenizer.token_to_id("[PAD]")], dtype=torch.int64)
    direction_token = torch.tensor(
        [src_tokenizer.token_to_id(f"[{direction.upper()}]")], 
        dtype=torch.int64
    )
    
    # Create encoder input
    enc_input_tokens = src_tokenizer.encode(text).ids[:seq_len-3]
    encoder_input = torch.cat([
        sos_token,
        direction_token,
        torch.tensor(enc_input_tokens, dtype=torch.int64),
        eos_token,
        torch.tensor([pad_token.item()] * (seq_len - len(enc_input_tokens) - 3), dtype=torch.int64)
    ]).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Initialize decoder input with SOS and padding
        decoder_input = torch.full((1, seq_len), 
                                 pad_token.item(),
                                 dtype=torch.long, 
                                 device=device)
        decoder_input[0, 0] = tgt_tokenizer.token_to_id("[SOS]")
        
        generated_tokens = []
        current_pos = 1  # Start after SOS token
        
        for _ in range(seq_len-1):
            outputs = model(encoder_input, decoder_input)
            next_token = outputs[0, current_pos-1].argmax()
            
            if next_token.item() == tgt_tokenizer.token_to_id("[EOS]"):
                break
            
            generated_tokens.append(next_token.item())
            decoder_input[0, current_pos] = next_token
            current_pos += 1
    
    # Convert tokens to text
    output_text = []
    for token_id in generated_tokens:
        token = tgt_tokenizer.id_to_token(token_id)
        if token == "[EOS]" or token == "[PAD]":
            break
        output_text.append(token)
    
    return " ".join(output_text)

def load_model_for_inference(model_path, device):
    """Load a trained model for inference"""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # Initialize model components
    encoder = Encoder(
        config['num_encoder_layers'],
        config['dim_model'],
        config['num_heads'],
        config['window_size'],
        config['dropout']
    )
    decoder = Decoder(
        config['num_decoder_layers'],
        config['dim_model'],
        config['num_heads'],
        config['window_size'],
        config['dropout']
    )
    source_embeddings = InputEmbeddings(config['dim_model'], config['vocab_size'])
    target_embeddings = InputEmbeddings(config['dim_model'], config['vocab_size'])
    source_positions = RotaryPositionalEncoding(config['dim_model'], config['seq_len'])
    target_positions = RotaryPositionalEncoding(config['dim_model'], config['seq_len'])
    
    model = Transformer(
        encoder, decoder,
        source_embeddings, target_embeddings,
        source_positions, target_positions,
        config['vocab_size']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, config

# Example usage:
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model and tokenizers
    model, config = load_model_for_inference(r'C:\Users\Cihan\Desktop\llamaindex\machine-translation\metrics-epoch3\epoch_2_best_model_fr2ru.pt', device)
    
    print("Loading dataset...")
    dataset = load_dataset("opus_books", f"{config['src_lang']}-{config['tgt_lang']}", split="train")
    
    # Create tokenizers
    print("Creating tokenizers...")
    fr_tokenizer, ru_tokenizer = create_tokenizers(dataset)
    
    # Example translations
    french_text = "chien"
    russian_text = "собака"
    
    # French to Russian
    print(f"\nFrench to Russian:")
    print(f"Input: {french_text}")
    ru_translation = translate(
        model, french_text, fr_tokenizer, ru_tokenizer, 
        config['seq_len'], device, "fr2ru"
    )
    print(type(ru_translation))
    print(len(ru_translation))
    print(f"Translation: {ru_translation}")
    
    # Russian to French
    print(f"\nRussian to French:")
    print(f"Input: {russian_text}")
    fr_translation = translate(
        model, russian_text, ru_tokenizer, fr_tokenizer, 
        config['seq_len'], device, "ru2fr"
    )
    print(type(fr_translation))
    print(len(fr_translation))
    print(f"Translation: {fr_translation}")