import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models
from typing import Dict, Tuple, List
import wandb
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

class BaseImageCaptioning(pl.LightningModule):
    def __init__(self, vocab_size: int, embed_size: int = 256, hidden_size: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        
        # CNN Encoder (VGG19)
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(vgg19.features.children()))
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
        
    def tokens_to_words(self, tokens: List[int]) -> str:
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
            ).squeeze().detach().cpu().numpy()
            
            # Normalize attention
            attn_range = attn.max() - attn.min()
            if attn_range > 0:
                attn = (attn - attn.min()) / attn_range
            else:
                attn = np.zeros_like(attn)  # If max=min, set attention to zeros
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
        data = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        
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
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, List[List[str]]], batch_idx: int) -> Dict:
        images, captions, original_captions = batch
        loss = self.forward(images, captions)
        
        # Log training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # In overfit mode, log the first batch images and captions
        trainer = self.trainer
        is_overfit = trainer.overfit_batches > 0
        
        if is_overfit and batch_idx == 0:
            generated_captions = self.generate_caption(images)
            self.log_images_and_captions(images, generated_captions, original_captions, prefix="train")
        
        return {'loss': loss}
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, List[List[str]]], batch_idx: int) -> Dict:
        images, captions, original_captions = batch
        loss = self.forward(images, captions)
        
        # Generate captions for BLEU score calculation
        generated_captions = self.generate_caption(images)
        
        # Log images and captions
        # In overfit mode, log the first batch every time
        # In normal mode, log every 100 batches
        trainer = self.trainer
        is_overfit = trainer.overfit_batches > 0
        
        if (is_overfit and batch_idx == 0) or (not is_overfit and batch_idx % 100 == 0):
            self.log_images_and_captions(images, generated_captions, original_captions, prefix="val")
        
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
        
        # Process generated captions and references
        hypotheses = []  # Generated captions
        references = []  # Ground truth captions
        
        for output in outputs:
            for gen_caption in output['generated_captions']:
                hypothesis = self.tokens_to_words(gen_caption)
                hypotheses.append(hypothesis)
                
            batch_references = []
            for caption_tensor in output['target_captions']:
                reference = self.tokens_to_words(caption_tensor.tolist())
                if reference:  # Only add non-empty references
                    batch_references.append([reference])
            references.extend(batch_references)
        
        # Calculate BLEU scores with smoothing
        if hypotheses and references:
            # Initialize smoothing function
            # NLTK's method1 adds a small constant (1) to both numerator and denominator of the modified precision calculation, which helps avoid zero scores when there are no matches.
            smoothing = SmoothingFunction().method1
            
            # Calculate BLEU scores with smoothing
            bleu1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0), smoothing_function=smoothing)
            bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
            bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
            bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
        else:
            bleu1 = bleu2 = bleu3 = bleu4 = 0.0
            
        # Log metrics
        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('bleu1', bleu1 * 100, prog_bar=True)  # Convert to percentage
        self.log('bleu2', bleu2 * 100, prog_bar=True)
        self.log('bleu3', bleu3 * 100, prog_bar=True)
        self.log('bleu4', bleu4 * 100, prog_bar=True)
        
        # Clear memory
        self.validation_step_outputs.clear()
    
    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """To be implemented by subclasses."""
        raise NotImplementedError
    
    def generate_caption(self, image: torch.Tensor) -> list:
        """To be implemented by subclasses."""
        raise NotImplementedError 
    
    def log_images_and_captions(self, images: torch.Tensor, generated_captions: list, original_captions: List[List[str]], 
                               attention_maps: torch.Tensor = None, max_images: int = 4, prefix: str = "val"):
        """Log images and their generated captions to wandb."""
        # Convert images back to [0,1] range for visualization
        images = self.inverse_transform(images)
        images = [img.cpu().numpy().transpose(1, 2, 0) for img in images]
        
        # Create wandb Image objects with captions
        wandb_images = []
        for idx, (image, gen_caption, orig_captions) in enumerate(zip(images, generated_captions, original_captions)):
            # Clip values to [0,1] range
            image = np.clip(image, 0, 1)
            
            # Format caption text with generated and original captions
            caption_text = f"Generated: {" ".join(self.tokens_to_words(gen_caption))}\n\nOriginal captions:\n"
            for i, orig_cap in enumerate(orig_captions, 1):
                caption_text += f"{i}. {orig_cap}\n"
            
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
                attn_range = attn.max() - attn.min()
                attn = (attn - attn.min()) / attn_range if attn_range > 0 else np.zeros_like(attn)
                attn_colored = np.stack([attn, attn, attn], axis=-1)
                
                # Blend attention map with original image
                alpha = 0.8
                blended = (1-alpha)*image + alpha*attn_colored
                blended = np.clip(blended, 0, 1)
                
                # Create attention grid for individual words
                attention_grid = self.create_attention_grid(image, gen_caption, attention_maps[idx])
                
                # Log both average attention and per-word attention grid
                wandb_images.extend([
                    wandb.Image(blended, caption=caption_text),
                    wandb.Image(attention_grid, caption=f"Word-by-word attention") if attention_grid is not None else None
                ])
            else:
                wandb_images.append(wandb.Image(image, caption=caption_text))
        
        # Log to wandb with prefix
        # Filter out None values from wandb_images
        wandb_images = [img for img in wandb_images if img is not None]
        wandb.log({f"{prefix}_samples": wandb_images}) 