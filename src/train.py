import os
import argparse
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
from datetime import datetime

from data.dataset import Flickr8kDataset, flickr8k_collate_fn
from models.show_and_tell import ShowAndTell
from models.show_attend_tell import ShowAttendTell

# Fixed data directory relative to project root
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def main(args):
    # Initialize wandb (enable for overfit, disable for fast debug)
    if args.debug != "fast":
        # Create a unique name with timestamp and model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{args.model}_{timestamp}"
        if args.debug == "overfit":
            run_name += "_overfit"
        wandb.init(project="image-captioning-comparison", name=run_name)
        wandb_logger = WandbLogger(name=run_name, version=run_name)
    else:
        wandb_logger = None
    
    # Data loading
    train_dataset = Flickr8kDataset(
        image_dir=os.path.join(DATA_DIR, 'Images'),
        captions_file=os.path.join(DATA_DIR, 'captions.txt'),
        split='train'
    )
    
    val_dataset = Flickr8kDataset(
        image_dir=os.path.join(DATA_DIR, 'Images'),
        captions_file=os.path.join(DATA_DIR, 'captions.txt'),
        split='val'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=flickr8k_collate_fn,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=flickr8k_collate_fn,
        persistent_workers=True
    )
    
    # Model
    if args.model == 'show_tell':
        model = ShowAndTell(
            vocab_size=len(train_dataset.vocab),
            embed_size=args.embed_size,
            hidden_size=args.hidden_size
        )
    else:
        model = ShowAttendTell(
            vocab_size=len(train_dataset.vocab),
            embed_size=args.embed_size,
            hidden_size=args.hidden_size,
            attention_dim=args.attention_dim
        )
    
    # Set vocabulary for caption conversion
    model.set_vocabulary(train_dataset.vocab)
    
    # Callbacks
    callbacks = []
    if not args.debug:
        # Save best models based on validation loss
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename=f'{args.model}-{{epoch:02d}}-{{val_loss:.2f}}',
            monitor='val_loss',
            mode='min',
            save_top_k=3
        )
        
        # Save last model
        last_model_callback = ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename=f'{args.model}-last',
            save_last=True
        )
        callbacks.extend([checkpoint_callback, last_model_callback])
    
    # Trainer configuration
    trainer_kwargs = {
        'max_epochs': args.epochs,
        'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
        'devices': 1,
        'callbacks': callbacks,
        'logger': wandb_logger,
        'gradient_clip_val': args.grad_clip,
        'default_root_dir': f"lightning_logs/{run_name}" if not args.debug else None
    }
    
    # Add debug configurations
    if args.debug == "overfit":
        trainer_kwargs.update({
            'overfit_batches': 1,  # Use 1 batch of training data
            'max_epochs': 1000,  # Increase epochs for overfitting
            'val_check_interval': 1.0  # Validate after each epoch
        })
    elif args.debug == "fast":
        trainer_kwargs.update({
            'fast_dev_run': True  # Run 1 train, val batch
        })
    
    trainer = pl.Trainer(**trainer_kwargs)
    
    # Train
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train image captioning models')
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True,
                        choices=['show_tell', 'show_attend_tell'],
                        help='Model architecture to use')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--embed_size', type=int, default=256,
                        help='Word embedding size')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='LSTM hidden size')
    parser.add_argument('--attention_dim', type=int, default=256,
                        help='Attention layer dimension (for Show, Attend and Tell)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--grad_clip', type=float, default=5.,
                        help='Gradient clipping value')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--debug', type=str, choices=['overfit', 'fast'],
                        help='Debug mode: "overfit" for overfitting test, "fast" for fast_dev_run')
    
    args = parser.parse_args()
    
    # Create checkpoint directory if it doesn't exist and not in debug mode
    if not args.debug:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    main(args) 