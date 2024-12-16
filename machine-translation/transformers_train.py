from datasets import load_dataset
from tqdm.auto import tqdm

import yaml
import json
from pathlib import Path
from datetime import datetime
import argparse

import torch
print(f"Using {torch.__version__=}")
torch.set_float32_matmul_precision('high')

from torch.utils.data import random_split, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


from transformers_model import (
    Transformer, Encoder, Decoder, 
    InputEmbeddings, RotaryPositionalEncoding
)
from transformers_dataset import (
    get_tokenizers,
    create_combined_dataset
)

class MetricsTracker:
    def __init__(self, save_dir=r"C:/Users/Cihan/Desktop/llamaindex/machine-translation/metrics"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.metrics = {
            "fr2ru": {
                "train_losses": [],
                "val_losses": []
            },
            "ru2fr": {
                "train_losses": [],
                "val_losses": []
            },
            "learning_rates": [],
            "epochs": []
        }
        
        self.filename = self.save_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    def update_direction(self, direction, train_loss=None, val_loss=None):
        if train_loss is not None:
            self.metrics[direction]["train_losses"].append(train_loss)
        if val_loss is not None:
            self.metrics[direction]["val_losses"].append(val_loss)
    
    def update_epoch(self, lr=None, epoch=None):
        if lr is not None:
            self.metrics["learning_rates"].append(lr)
        if epoch is not None:
            self.metrics["epochs"].append(epoch)
    
    def save(self):
        with open(self.filename, 'w') as f:
            json.dump(self.metrics, f, indent=4)

def analyze_combined_dataset_lengths(combined_dataset):
    """
    Analyze sequence lengths in the CombinedBilingualDataset for both translation directions.
    
    Args:
        combined_dataset: CombinedBilingualDataset instance
    
    Returns:
        dict: Statistics about sequence lengths and recommended parameters
    """
    fr2ru_enc_lengths = []
    fr2ru_dec_lengths = []
    ru2fr_enc_lengths = []
    ru2fr_dec_lengths = []
    
    print("Analyzing sequence lengths in processed dataset...")
    for idx in tqdm(range(len(combined_dataset))):
        item = combined_dataset[idx]
        
        # Get actual lengths (excluding padding)
        fr2ru_enc = item['fr2ru']['encoder_input']
        fr2ru_dec = item['fr2ru']['decoder_input']
        ru2fr_enc = item['ru2fr']['encoder_input']
        ru2fr_dec = item['ru2fr']['decoder_input']
        
        # Count non-padding tokens
        fr2ru_enc_len = (fr2ru_enc != combined_dataset.pad_token).sum().item()
        fr2ru_dec_len = (fr2ru_dec != combined_dataset.pad_token).sum().item()
        ru2fr_enc_len = (ru2fr_enc != combined_dataset.pad_token).sum().item()
        ru2fr_dec_len = (ru2fr_dec != combined_dataset.pad_token).sum().item()
        
        fr2ru_enc_lengths.append(fr2ru_enc_len)
        fr2ru_dec_lengths.append(fr2ru_dec_len)
        ru2fr_enc_lengths.append(ru2fr_enc_len)
        ru2fr_dec_lengths.append(ru2fr_dec_len)
    
    stats = {
        'fr2ru': {
            'encoder': calculate_stats(fr2ru_enc_lengths),
            'decoder': calculate_stats(fr2ru_dec_lengths)
        },
        'ru2fr': {
            'encoder': calculate_stats(ru2fr_enc_lengths),
            'decoder': calculate_stats(ru2fr_dec_lengths)
        }
    }
    
    # Find the maximum required sequence length
    max_95th = max(
        stats['fr2ru']['encoder']['p95'],
        stats['fr2ru']['decoder']['p95'],
        stats['ru2fr']['encoder']['p95'],
        stats['ru2fr']['decoder']['p95']
    )
    
    recommended_seq_len = int(max_95th) + 2  # Add some buffer for safety
    
    # Calculate coverage for recommended length
    stats['recommendations'] = {
        'seq_len': recommended_seq_len,
        'coverage': {
            'fr2ru_enc': calculate_coverage(fr2ru_enc_lengths, recommended_seq_len),
            'fr2ru_dec': calculate_coverage(fr2ru_dec_lengths, recommended_seq_len),
            'ru2fr_enc': calculate_coverage(ru2fr_enc_lengths, recommended_seq_len),
            'ru2fr_dec': calculate_coverage(ru2fr_dec_lengths, recommended_seq_len)
        }
    }
    
    return stats

def calculate_stats(lengths):
    """Calculate statistics for a list of lengths."""
    return {
        'mean': np.mean(lengths),
        'median': np.median(lengths),
        'p90': np.percentile(lengths, 90),
        'p95': np.percentile(lengths, 95),
        'p99': np.percentile(lengths, 99),
        'max': max(lengths),
        'min': min(lengths)
    }

def calculate_coverage(lengths, seq_len):
    """Calculate percentage of sequences covered by given sequence length."""
    return (np.array(lengths) <= seq_len).mean() * 100

def print_sequence_analysis(stats):
    """Print formatted analysis results."""
    print("\nSequence Length Analysis:")
    
    for direction in ['fr2ru', 'ru2fr']:
        print(f"\n{direction.upper()} Direction:")
        print("  Encoder Input:")
        print(f"    Mean length: {stats[direction]['encoder']['mean']:.1f} tokens")
        print(f"    Median length: {stats[direction]['encoder']['median']:.1f} tokens")
        print(f"    95th percentile: {stats[direction]['encoder']['p95']:.1f} tokens")
        print(f"    Range: {stats[direction]['encoder']['min']} - {stats[direction]['encoder']['max']} tokens")
        
        print("  Decoder Input:")
        print(f"    Mean length: {stats[direction]['decoder']['mean']:.1f} tokens")
        print(f"    Median length: {stats[direction]['decoder']['median']:.1f} tokens")
        print(f"    95th percentile: {stats[direction]['decoder']['p95']:.1f} tokens")
        print(f"    Range: {stats[direction]['decoder']['min']} - {stats[direction]['decoder']['max']} tokens")
    
    print("\nRecommendations:")
    print(f"  Recommended sequence length: {stats['recommendations']['seq_len']}")
    print("\nCoverage with recommended length:")
    print(f"  FR→RU Encoder: {stats['recommendations']['coverage']['fr2ru_enc']:.1f}%")
    print(f"  FR→RU Decoder: {stats['recommendations']['coverage']['fr2ru_dec']:.1f}%")
    print(f"  RU→FR Encoder: {stats['recommendations']['coverage']['ru2fr_enc']:.1f}%")
    print(f"  RU→FR Decoder: {stats['recommendations']['coverage']['ru2fr_dec']:.1f}%")

def create_dataloaders(ds_raw, tokenizer_fr, tokenizer_ru, seq_len, batch_size, train_size: float = 0.9, num_workers: int = 4):
    """Create train and validation dataloaders."""

    # Split dataset
    train_size = int(train_size * len(ds_raw))
    val_size = len(ds_raw) - train_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_size, val_size])
    
    # Create datasets
    train_ds = create_combined_dataset(
        ds=train_ds_raw,
        tokenizer_fr=tokenizer_fr,
        tokenizer_ru=tokenizer_ru,
        seq_len=seq_len
    )
    
    val_ds = create_combined_dataset(
        ds=val_ds_raw,
        tokenizer_fr=tokenizer_fr,
        tokenizer_ru=tokenizer_ru,
        seq_len=seq_len
    )
    
    train_dataloader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_dataloader, val_dataloader

def create_or_load_model(config, device, resume_checkpoint=None):
    """Initialize model and optionally load from checkpoint."""
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
    ).to(device)
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(
        model.parameters(), 
        lr=float(config['learning_rate']), 
        weight_decay=config['weight_decay']
    )
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=config['steps_per_epoch'] * config['num_epochs']
    )
    
    # Initialize training state
    start_epoch = 0
    best_val_losses = {"fr2ru": float('inf'), "ru2fr": float('inf')}
    
    # Load checkpoint if provided
    if resume_checkpoint:
        print(f"Loading checkpoint from {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        
        # Validate config compatibility
        saved_config = checkpoint['config']
        for key in ['dim_model', 'num_heads', 'vocab_size']:
            if saved_config.get(key) != config.get(key):
                raise ValueError(
                    f"Configuration mismatch for {key}: "
                    f"saved={saved_config.get(key)}, "
                    f"provided={config.get(key)}"
                )
        
        # Load states
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore training state
        start_epoch = checkpoint['epoch'] + 1
        best_val_losses[checkpoint['direction']] = checkpoint['val_loss']
        
        print(f"Resumed from epoch {start_epoch}")
        print(f"Best validation loss ({checkpoint['direction']}): {checkpoint['val_loss']:.4f}")
    
    return {
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'start_epoch': start_epoch,
        'best_val_losses': best_val_losses
    }


def validate(model, val_loader, criterion, device):
    model.eval()
    total_losses = {"fr2ru": 0, "ru2fr": 0}
    total_steps = {"fr2ru": 0, "ru2fr": 0}
    
    with torch.no_grad():
        for batch in val_loader:
            for direction in ['fr2ru', 'ru2fr']:
                encoder_input = batch[direction]['encoder_input'].to(device)
                decoder_input = batch[direction]['decoder_input'].to(device)
                labels = batch[direction]['label'].to(device)
                
                outputs = model(encoder_input, decoder_input)
                outputs = outputs.view(-1, outputs.size(-1))
                labels = labels.view(-1)
                
                loss = criterion(outputs, labels)
                total_losses[direction] += loss.item()
                total_steps[direction] += 1
    
    return {
        "fr2ru": total_losses["fr2ru"] / total_steps["fr2ru"],
        "ru2fr": total_losses["ru2fr"] / total_steps["ru2fr"]
    }

def train(config_path, resume_checkpoint=None):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("opus_books", f"{config['src_lang']}-{config['tgt_lang']}", split="train")
    
    # Create tokenizers
    print("Creating tokenizers...")
    fr_tokenizer, ru_tokenizer = get_tokenizers(dataset)
    
    # Update vocab size based on actual tokenizer vocabulary
    vocab_size = max(len(fr_tokenizer.vocab), len(ru_tokenizer.vocab))
    print(f"Vocabulary size: {vocab_size}")
    config['vocab_size'] = vocab_size
    
    # Split dataset
    train_size = int(config['train_test_split'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create combined datasets
    print("Creating datasets...")
    train_data = create_combined_dataset(
        train_dataset, fr_tokenizer, ru_tokenizer, config['seq_len']
    )
    val_data = create_combined_dataset(
        val_dataset, fr_tokenizer, ru_tokenizer, config['seq_len']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_data, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers']
    )
    val_loader = DataLoader(
        val_data, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers']
    )
    
    # Initialize model
    print("Initializing model...")
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
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Running on {device}")
    model = Transformer(
        encoder, decoder,
        source_embeddings, target_embeddings,
        source_positions, target_positions,
        config['vocab_size']
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    #model = torch.compile(model)

    # Initialize optimizer, scheduler, criterion and metrics tracker
    optimizer = AdamW(
        model.parameters(), 
        lr=float(config['learning_rate']), 
        weight_decay=config['weight_decay']
    )
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=len(train_loader) * config['num_epochs']
    )
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)  # ignore padding token

    metrics = MetricsTracker()
    
    best_val_losses = {"fr2ru": float('inf'), "ru2fr": float('inf')}
    start_epoch = 0
    

    # Save the configuration along with the metrics
    with open(metrics.save_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    if resume_checkpoint is not None:
        print(f"Loading checkpoint from {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load other training state
        start_epoch = checkpoint['epoch'] + 1
        
        print(f"Resuming from epoch {start_epoch}")
    

    print("Starting training...")
    for epoch in range(start_epoch, start_epoch+config['num_epochs']):
        model.train()
        total_losses = {"fr2ru": 0, "ru2fr": 0}
        total_steps = {"fr2ru": 0, "ru2fr": 0}
        
        # Training loop
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{start_epoch+ config["num_epochs"]}')
        for batch in progress_bar:
            # Process both directions
            for direction in ['fr2ru', 'ru2fr']:
                optimizer.zero_grad()
                
                # Get inputs for current direction
                encoder_input = batch[direction]['encoder_input'].to(device)
                decoder_input = batch[direction]['decoder_input'].to(device)
                labels = batch[direction]['label'].to(device)
                
                # Forward pass
                
                outputs = model(encoder_input, decoder_input)
                
                # Reshape outputs and labels for loss computation
                outputs = outputs.view(-1, outputs.size(-1))
                labels = labels.view(-1)
                
                # Compute loss
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Clip gradients
                if config['grad_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                
                optimizer.step()
                scheduler.step()
                
                # Update metrics
                total_losses[direction] += loss.item()
                total_steps[direction] += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    f'{direction}_loss': f'{loss.item():.4f}',
                    f'{direction}_avg': f'{total_losses[direction]/total_steps[direction]:.4f}'
                })
        
        # Calculate average training loss for the epoch
        avg_train_losses = {
            direction: total_losses[direction] / total_steps[direction]
            for direction in ['fr2ru', 'ru2fr']
        }
        
        # Validation
        val_losses = validate(model, val_loader, criterion, device)
        print(f"\nEpoch {epoch+1}/{start_epoch+config['num_epochs']}")
        for direction in ['fr2ru', 'ru2fr']:
            print(f"{direction.upper()}:")
            print(f"  Train Loss: {avg_train_losses[direction]:.4f}")
            print(f"  Val Loss: {val_losses[direction]:.4f}")
        
        # Update and save metrics
        for direction in ['fr2ru', 'ru2fr']:
            metrics.update_direction(
                direction,
                train_loss=avg_train_losses[direction],
                val_loss=val_losses[direction]
            )
        metrics.update_epoch(lr=scheduler.get_last_lr()[0], epoch=epoch)
        metrics.save()
        
        # Save best model
        for direction in ['fr2ru', 'ru2fr']:
            if val_losses[direction] < best_val_losses[direction]:
                best_val_losses[direction] = val_losses[direction]
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': avg_train_losses[direction],
                    'val_loss': val_losses[direction],
                    'direction': direction,
                    'config': config
                }, metrics.save_dir / f'epoch_{epoch}_best_model_{direction}.pt')

import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the translation model')
    parser.add_argument('--config', type=str, default=r'C:/Users/Cihan/Desktop/llamaindex/machine-translation/config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    train(args.config)