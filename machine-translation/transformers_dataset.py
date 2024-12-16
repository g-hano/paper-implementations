import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from spacy.lang.fr import French
from spacy.lang.ru import Russian

from tqdm import tqdm
import json
from pathlib import Path

class SpacyTokenizerWrapper:
    def __init__(self, tokenizer):
        """
        Wrapper for spaCy tokenizer that adds vocabulary and special token handling.
        
        Args:
            tokenizer: spaCy tokenizer instance
        """
        self.tokenizer = tokenizer
        
        # Create vocabulary with special tokens
        self.special_tokens = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[SOS]": 2,
            "[EOS]": 3,
            "[FR2RU]": 4,
            "[RU2FR]": 5,
        }
        
        # Initialize vocabulary with special tokens
        self.vocab = self.special_tokens.copy()
        self.vocab_reverse = {v: k for k, v in self.vocab.items()}
        self.next_token_id = len(self.special_tokens)

    def token_to_id(self, token: str) -> int:
        """Get token id for a given token string."""
        return self.vocab.get(token, self.special_tokens["[UNK]"])
    
    def id_to_token(self, id: int) -> str:
        """Get token string for a given token id."""
        return self.vocab_reverse.get(id, "[UNK]")
    
    def encode(self, text: str):
        """Encode text to token ids."""
        tokens = [token.text for token in self.tokenizer(text)]
        ids = []
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = self.next_token_id
                self.vocab_reverse[self.next_token_id] = token
                self.next_token_id += 1
            ids.append(self.vocab[token])
        return type('TokenizerOutput', (), {'ids': ids})()
    
    def save(self, path: str):
        """Save tokenizer vocabulary to JSON file."""
        save_dict = {
            'vocab': self.vocab,
            'vocab_reverse': self.vocab_reverse,
            'next_token_id': self.next_token_id,
            'special_tokens': self.special_tokens
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(save_dict, f, ensure_ascii=False, indent=2)
            
    @classmethod
    def load(cls, path: str, spacy_tokenizer):
        """Load tokenizer vocabulary from JSON file."""
        instance = cls(spacy_tokenizer)
        
        with open(path, 'r', encoding='utf-8') as f:
            save_dict = json.load(f)
            
        instance.vocab = save_dict['vocab']
        instance.vocab_reverse = save_dict['vocab_reverse']
        instance.next_token_id = save_dict['next_token_id']
        instance.special_tokens = save_dict['special_tokens']
        
        return instance

class CombinedBilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_fr, tokenizer_ru, seq_len):
        """
        Initialize a dataset that handles both translation directions in a single pass.
        Each item will contain both fr→ru and ru→fr translations.
        """
        super().__init__()
        self.ds = ds
        self.seq_len = seq_len
        self.tokenizer_fr = tokenizer_fr
        self.tokenizer_ru = tokenizer_ru
        
        # Special tokens - we'll use the same special tokens for both directions
        self.sos_token = torch.tensor([tokenizer_fr.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_fr.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_fr.token_to_id("[PAD]")], dtype=torch.int64)
        
        # Direction tokens
        self.fr2ru_token = torch.tensor([tokenizer_fr.token_to_id("[FR2RU]")], dtype=torch.int64)
        self.ru2fr_token = torch.tensor([tokenizer_fr.token_to_id("[RU2FR]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def process_direction(self, src_text, tgt_text, src_tokenizer, tgt_tokenizer, direction_token):
        """Helper function to process one direction of translation"""
        # Transform the text into tokens
        enc_input_tokens = src_tokenizer.encode(src_text).ids
        dec_input_tokens = tgt_tokenizer.encode(tgt_text).ids

        # Truncate if sequences are too long
        max_enc_len = self.seq_len - 3  # -3 for SOS, direction token, and EOS
        max_dec_len = self.seq_len - 1  # -1 for SOS
        
        enc_input_tokens = enc_input_tokens[:max_enc_len]
        dec_input_tokens = dec_input_tokens[:max_dec_len]
        
        # Calculate padding
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 3
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Create tensors
        encoder_input = torch.cat(
            [
                self.sos_token,
                direction_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        return encoder_input, decoder_input, label

    def __getitem__(self, idx):
        pair = self.ds[idx]
        fr_text = pair['translation']['fr']
        ru_text = pair['translation']['ru']

        # Process fr→ru direction
        fr2ru_encoder, fr2ru_decoder, fr2ru_label = self.process_direction(
            fr_text, ru_text,
            self.tokenizer_fr, self.tokenizer_ru,
            self.fr2ru_token
        )

        # Process ru→fr direction
        ru2fr_encoder, ru2fr_decoder, ru2fr_label = self.process_direction(
            ru_text, fr_text,
            self.tokenizer_ru, self.tokenizer_fr,
            self.ru2fr_token
        )

        return {
            # fr→ru direction
            "fr2ru": {
                "encoder_input": fr2ru_encoder,  # (seq_len)
                "decoder_input": fr2ru_decoder,  # (seq_len)
                "label": fr2ru_label,  # (seq_len)
                "src_text": fr_text,
                "tgt_text": ru_text,
                "src_lang": 'fr',
                "tgt_lang": 'ru',
                "direction": "fr2ru"
            },
            # ru→fr direction
            "ru2fr": {
                "encoder_input": ru2fr_encoder,  # (seq_len)
                "decoder_input": ru2fr_decoder,  # (seq_len)
                "label": ru2fr_label,  # (seq_len)
                "src_text": ru_text,
                "tgt_text": fr_text,
                "src_lang": 'ru',
                "tgt_lang": 'fr',
                "direction": "ru2fr"
            }
        }
    
def create_tokenizers(dataset):
    fr = French()
    ru = Russian()
    fr_tokenizer = SpacyTokenizerWrapper(fr.tokenizer)
    ru_tokenizer = SpacyTokenizerWrapper(ru.tokenizer)

    # Build vocabularies
    print("Building vocabularies...")
    for item in tqdm(dataset):
        # Process French text
        fr_tokenizer.encode(item['translation']['fr'])
        # Process Russian text
        ru_tokenizer.encode(item['translation']['ru'])
        
    print(f"French vocabulary size: {len(fr_tokenizer.vocab)}")
    print(f"Russian vocabulary size: {len(ru_tokenizer.vocab)}")
    
    return fr_tokenizer, ru_tokenizer

def create_combined_dataset(ds, tokenizer_fr, tokenizer_ru, seq_len=128):
    """Create a single dataset that handles both translation directions."""
    return CombinedBilingualDataset(
        ds=ds,
        tokenizer_fr=tokenizer_fr,
        tokenizer_ru=tokenizer_ru,
        seq_len=seq_len
    )

def get_tokenizers(dataset=None, force_rebuild=False):
    """
    Create new tokenizers or load existing ones.
    
    Args:
        dataset: Dataset to build vocabulary from (optional if loading from file)
        force_rebuild: If True, rebuilds tokenizers even if saved files exist
    """
    fr_path = r'C:/Users/Cihan/Desktop/llamaindex/machine-translation/tokenizers/fr_tokenizer.json'
    ru_path = r'C:/Users/Cihan/Desktop/llamaindex/machine-translation/tokenizers/ru_tokenizer.json'
    
    if not force_rebuild and Path(fr_path).exists() and Path(ru_path).exists():
        print("Loading existing tokenizers...")
        fr = French()
        ru = Russian()
        fr_tokenizer = SpacyTokenizerWrapper.load(fr_path, fr.tokenizer)
        ru_tokenizer = SpacyTokenizerWrapper.load(ru_path, ru.tokenizer)
        
        print(f"Loaded French vocabulary size: {len(fr_tokenizer.vocab)}")
        print(f"Loaded Russian vocabulary size: {len(ru_tokenizer.vocab)}")
        
        return fr_tokenizer, ru_tokenizer
    
    elif dataset is None:
        raise ValueError("Dataset is required when building new tokenizers")
    
    else:
        print("Building new tokenizers...")
        fr = French()
        ru = Russian()
        fr_tokenizer = SpacyTokenizerWrapper(fr.tokenizer)
        ru_tokenizer = SpacyTokenizerWrapper(ru.tokenizer)

        # Build vocabularies
        print("Building vocabularies...")
        for item in tqdm(dataset):
            # Process French text
            fr_tokenizer.encode(item['translation']['fr'])
            # Process Russian text
            ru_tokenizer.encode(item['translation']['ru'])
            
        print(f"French vocabulary size: {len(fr_tokenizer.vocab)}")
        print(f"Russian vocabulary size: {len(ru_tokenizer.vocab)}")
        
        # Save tokenizers
        print("Saving tokenizers...")
        fr_tokenizer.save(fr_path)
        ru_tokenizer.save(ru_path)
        
        return fr_tokenizer, ru_tokenizer

#dataset = load_dataset("opus_books", "fr-ru", split="all")
#fr_tokenizer, ru_tokenizer = create_tokenizers(dataset)
#fr_tokenizer, ru_tokenizer = get_tokenizers(dataset=dataset)
#combined = create_combined_dataset(dataset, fr_tokenizer, ru_tokenizer, 250)
