from transformers_model import (
    Transformer, Encoder, Decoder, 
    InputEmbeddings, RotaryPositionalEncoding
)
import torch
import torch.nn.functional as F

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

def translate_with_beam_search(model, text, src_tokenizer, tgt_tokenizer, seq_len, device, 
                             direction="fr2ru", beam_width=5, length_penalty=0.6, max_length=None):
    """
    Translate text using beam search decoding.
    
    Args:
        model: The transformer model
        text: Input text to translate
        src_tokenizer: Source language tokenizer
        tgt_tokenizer: Target language tokenizer
        seq_len: Maximum sequence length
        device: Device to run the model on
        direction: Translation direction ("fr2ru" or "ru2fr")
        beam_width: Beam size for beam search
        length_penalty: Penalty factor for longer sequences
        max_length: Maximum generation length (defaults to seq_len if None)
    
    Returns:
        str: Best translation based on beam search
    """
    if max_length is None:
        max_length = seq_len
        
    model.eval()
    
    # Prepare special tokens
    sos_token = torch.tensor([src_tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
    eos_token = torch.tensor([src_tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
    pad_token = torch.tensor([src_tokenizer.token_to_id("[PAD]")], dtype=torch.int64)
    direction_token = torch.tensor(
        [src_tokenizer.token_to_id(f"[{direction.upper()}]")], 
        dtype=torch.int64
    )
    
    # Create and encode input sequence
    enc_input_tokens = src_tokenizer.encode(text).ids[:seq_len-3]
    encoder_input = torch.cat([
        sos_token,
        direction_token,
        torch.tensor(enc_input_tokens, dtype=torch.int64),
        eos_token,
        torch.tensor([pad_token.item()] * (seq_len - len(enc_input_tokens) - 3), dtype=torch.int64)
    ]).unsqueeze(0).to(device)
    
    # Initialize beam
    class Beam:
        def __init__(self, tokens, score, finished=False):
            self.tokens = tokens  # List of token ids
            self.score = score    # Log probability score
            self.finished = finished  # Whether beam reached EOS
            
        def __lt__(self, other):
            # For sorting beams
            return self.score > other.score
    
    with torch.no_grad():
        # Start with SOS token
        initial_beam = Beam(
            tokens=[tgt_tokenizer.token_to_id("[SOS]")],
            score=0.0,
            finished=False
        )
        beams = [initial_beam]
        
        # Generate tokens
        for step in range(max_length - 1):
            candidates = []
            
            # Process each beam
            for beam in beams:
                if beam.finished:
                    candidates.append(beam)
                    continue
                
                # Create decoder input from current beam
                decoder_input = torch.full(
                    (1, seq_len), 
                    pad_token.item(),
                    dtype=torch.long, 
                    device=device
                )
                for i, token_id in enumerate(beam.tokens):
                    decoder_input[0, i] = token_id
                
                # Get model predictions
                outputs = model(encoder_input, decoder_input)
                logits = outputs[0, len(beam.tokens)-1]
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Get top-k candidates
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)
                
                # Create new candidates
                for log_prob, token_id in zip(topk_log_probs, topk_indices):
                    token_id = token_id.item()
                    new_score = beam.score + log_prob.item()
                    
                    # Apply length penalty
                    length_norm = ((5 + len(beam.tokens)) ** length_penalty) / (6 ** length_penalty)
                    normalized_score = new_score / length_norm
                    
                    new_tokens = beam.tokens + [token_id]
                    is_finished = token_id == tgt_tokenizer.token_to_id("[EOS]")
                    
                    candidates.append(Beam(
                        tokens=new_tokens,
                        score=normalized_score,
                        finished=is_finished
                    ))
            
            # Select top beams for next step
            beams = sorted(candidates)[:beam_width]
            
            # Check if all beams are finished
            if all(beam.finished for beam in beams):
                break
        
        # Select best completed beam
        completed_beams = [beam for beam in beams if beam.finished]
        if completed_beams:
            best_beam = max(completed_beams, key=lambda x: x.score)
        else:
            best_beam = max(beams, key=lambda x: x.score)
        
        # Convert tokens to text
        output_tokens = []
        for token_id in best_beam.tokens[1:]:  # Skip SOS token
            token = tgt_tokenizer.id_to_token(token_id)
            if token == "[EOS]" or token == "[PAD]":
                break
            output_tokens.append(token)
        
        return " ".join(output_tokens)

# Example usage with different configurations
def translate_with_configs(model, text, src_tokenizer, tgt_tokenizer, seq_len, device, 
                         direction="fr2ru", configs=None):
    """
    Translate text with different beam search configurations.
    
    Args:
        model: The transformer model
        text: Input text to translate
        src_tokenizer, tgt_tokenizer: Source and target language tokenizers
        seq_len: Maximum sequence length
        device: Device to run the model on
        direction: Translation direction ("fr2ru" or "ru2fr")
        configs: Dictionary of beam search configurations to try
    
    Returns:
        dict: Translations for each configuration
    """
    if configs is None:
        configs = {
            'narrow_beam': {'beam_width': 3, 'length_penalty': 0.6},
            'wide_beam': {'beam_width': 10, 'length_penalty': 0.6},
            'length_penalized': {'beam_width': 5, 'length_penalty': 1.0},
            'length_favored': {'beam_width': 5, 'length_penalty': 0.3},
        }
    
    results = {}
    for name, config in configs.items():
        translation = translate_with_beam_search(
            model, text, src_tokenizer, tgt_tokenizer, seq_len, device,
            direction=direction, **config
        )
        results[name] = translation
    
    return results
