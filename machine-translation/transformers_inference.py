from utils import load_model_for_inference, translate_with_configs
from transformers_dataset import create_tokenizers
import torch
from datasets import load_dataset

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")
    model, config = load_model_for_inference(r'best_model.pt', device)
    
    print("Loading dataset...")
    dataset = load_dataset("opus_books", f"{config['src_lang']}-{config['tgt_lang']}", split="train")
    
    print("Creating tokenizers...")
    fr_tokenizer, ru_tokenizer = create_tokenizers(dataset)
    
    french_text = "chien"
    russian_text = "собака"
    
    # French to Russian
    print(f"\nFrench to Russian:")
    print(f"Input: {french_text}")
    ru_translation = translate_with_configs(
        model, french_text, fr_tokenizer, ru_tokenizer, 
        config['seq_len'], device, "fr2ru"
    )
    print(f"Translation: {ru_translation}")
    
    # Russian to French
    print(f"\nRussian to French:")
    print(f"Input: {russian_text}")
    fr_translation = translate_with_configs(
        model, russian_text, ru_tokenizer, fr_tokenizer, 
        config['seq_len'], device, "ru2fr"
    )
    print(f"Translation: {fr_translation}")
