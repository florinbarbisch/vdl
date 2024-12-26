import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models
from typing import Dict, Tuple
import wandb
from nltk.translate.bleu_score import corpus_bleu

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
        
        return {'loss': loss}
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict:
        images, captions = batch
        loss = self.forward(images, captions)
        
        # Generate captions for BLEU score calculation
        generated_captions = self.generate_caption(images)
        
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