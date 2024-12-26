import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models
from typing import Dict, Tuple, List
import wandb
from nltk.translate.bleu_score import corpus_bleu
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

class BaseImageCaptioning(pl.LightningModule):
    def __init__(self, vocab_size: int, embed_size: int = 256, hidden_size: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        
        # CNN Encoder (VGG16)
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(vgg16.features.children()))
        self.fc = nn.Linear(512 * 7 * 7, embed_size)  # 224x224 input -> 7x7 feature maps
        
        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Word embedding layer
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Initialize lists to store step outputs
        self.validation_step_outputs = []
        
        # Inverse transform for visualization
        self.inverse_transform = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.],
                              std=[1/0.229, 1/0.224, 1/0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                              std=[1., 1., 1.]),
        ])
        
        # Initialize vocabulary (to be set by dataset)
        self.id_to_word = None
        
    def set_vocabulary(self, vocab: dict):
        """Set the vocabulary mapping."""
        self.id_to_word = {v: k for k, v in vocab.items()}
        
    def tokens_to_words(self, tokens: List[int]) -> List[str]:
        """Convert token IDs to words."""
        if self.id_to_word is None:
            raise ValueError("Vocabulary not set. Call set_vocabulary first.")
            
        words = []
        for token in tokens:
            if token == 2:  # <end> token
                break
            if token >= 4:  # Skip special tokens
                words.append(self.id_to_word[token])
        return words
        
    def create_attention_grid(self, image: np.ndarray, caption: List[int], attention_maps: torch.Tensor) -> np.ndarray:
        """Create a grid of attention maps for each word."""
        words = self.tokens_to_words(caption)
        n_words = len(words)
        
        if n_words == 0:
            return None
            
        # Calculate grid dimensions
        n_cols = min(5, n_words)
        n_rows = (n_words + n_cols - 1) // n_cols
        
        # Create figure
        fig = plt.figure(figsize=(4*n_cols, 4*n_rows))
        
        for idx, (word, attn) in enumerate(zip(words, attention_maps[:n_words])):
            # Process attention map
            attn = attn.reshape(7, 7)
            attn = torch.nn.functional.interpolate(
                attn.unsqueeze(0).unsqueeze(0),
                size=image.shape[:2],
                mode='bilinear'
            ).squeeze().cpu().numpy()
            
            # Normalize attention
            attn = (attn - attn.min()) / (attn.max() - attn.min())
            attn_colored = np.stack([attn, attn, attn], axis=-1)
            
            # Blend with original image
            alpha = 0.7
            blended = (1-alpha)*image + alpha*attn_colored
            blended = np.clip(blended, 0, 1)
            
            # Add to grid
            plt.subplot(n_rows, n_cols, idx + 1)
            plt.imshow(blended)
            plt.title(word, fontsize=12)
            plt.axis('off')
        
        # Adjust layout and convert to numpy array
        plt.tight_layout()
        fig.canvas.draw()
        
        # Convert figure to numpy array
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return data / 255.0
        
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Extract image features using CNN encoder."""
        features = self.encoder(images)  # (batch_size, 512, 7, 7)
        features = features.view(features.size(0), -1)  # (batch_size, 512*7*7)
        features = self.fc(features)  # (batch_size, embed_size)
        return features
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()))
        return optimizer
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict:
        images, captions = batch
        loss = self.forward(images, captions)
        
        # Log training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # In overfit mode, log the first batch images and captions
        trainer = self.trainer
        is_overfit = trainer.overfit_batches > 0
        
        if is_overfit and batch_idx == 0:
            generated_captions = self.generate_caption(images)
            self.log_images_and_captions(images, generated_captions, prefix="train")
        
        return {'loss': loss}
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict:
        images, captions = batch
        loss = self.forward(images, captions)
        
        # Generate captions for BLEU score calculation
        generated_captions = self.generate_caption(images)
        
        # Log images and captions
        # In overfit mode, log the first batch every time
        # In normal mode, log every 100 batches
        trainer = self.trainer
        is_overfit = trainer.overfit_batches > 0
        
        if (is_overfit and batch_idx == 0) or (not is_overfit and batch_idx % 100 == 0):
            self.log_images_and_captions(images, generated_captions, prefix="val")
        
        # Store outputs for epoch end
        self.validation_step_outputs.append({
            'val_loss': loss,
            'generated_captions': generated_captions,
            'target_captions': captions
        })
        
        return {'val_loss': loss}
    
    def on_validation_epoch_end(self) -> None:
        outputs = self.validation_step_outputs
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        
        # Calculate BLEU score
        # Pad sequences to same length
        max_gen_len = max(len(x['generated_captions'][0]) for x in outputs)
        generated = []
        for x in outputs:
            padded = x['generated_captions'][0] + [0] * (max_gen_len - len(x['generated_captions'][0]))
            generated.append(padded)
        generated = torch.tensor(generated)
        targets = [x['target_captions'].tolist() for x in outputs]
        bleu_score = corpus_bleu(targets, generated)
        # Log metrics
        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('bleu_score', bleu_score, prog_bar=True)
        
        # Clear memory
        self.validation_step_outputs.clear()
    
    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """To be implemented by subclasses."""
        raise NotImplementedError
    
    def generate_caption(self, image: torch.Tensor) -> list:
        """To be implemented by subclasses."""
        raise NotImplementedError 
    
    def log_images_and_captions(self, images: torch.Tensor, captions: list, attention_maps: torch.Tensor = None, 
                               max_images: int = 4, prefix: str = "val"):
        """Log images and their generated captions to wandb."""
        if not self.logger:
            return
            
        # Convert images back to [0,1] range for visualization
        images = self.inverse_transform(images[:max_images])
        images = [img.cpu().numpy().transpose(1, 2, 0) for img in images]
        
        # Create wandb Image objects with captions
        wandb_images = []
        for idx, (image, caption) in enumerate(zip(images, captions[:max_images])):
            # Clip values to [0,1] range
            image = np.clip(image, 0, 1)
            
            if attention_maps is not None:
                # Log average attention map
                attn = attention_maps[idx].mean(0)  # Average attention across words
                attn = attn.reshape(7, 7)  # Reshape to spatial dimensions
                attn = torch.nn.functional.interpolate(
                    attn.unsqueeze(0).unsqueeze(0), 
                    size=image.shape[:2], 
                    mode='bilinear'
                ).squeeze().detach().cpu().numpy()
                
                # Create grayscale heatmap for average attention
                attn = (attn - attn.min()) / (attn.max() - attn.min())
                attn_colored = np.stack([attn, attn, attn], axis=-1)
                
                # Blend attention map with original image
                alpha = 0.8
                blended = (1-alpha)*image + alpha*attn_colored
                blended = np.clip(blended, 0, 1)
                
                # Create attention grid for individual words
                attention_grid = self.create_attention_grid(image, caption, attention_maps[idx])
                
                # Log both average attention and per-word attention grid
                wandb_images.extend([
                    wandb.Image(blended, caption=f"Generated (average attention): {self.tokens_to_words(caption)}"),
                    wandb.Image(attention_grid, caption=f"Word-by-word attention") if attention_grid is not None else None
                ])
            else:
                wandb_images.append(wandb.Image(image, caption=f"Generated: {self.tokens_to_words(caption)}"))
        
        # Log to wandb with prefix
        self.logger.experiment.log({f"{prefix}_samples": wandb_images}) 