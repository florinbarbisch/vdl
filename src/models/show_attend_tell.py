import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union
from .base_model import BaseImageCaptioning

class Attention(nn.Module):
    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        
    def forward(self, encoder_out: torch.Tensor, decoder_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for attention mechanism.
        
        Args:
            encoder_out: Image features (batch_size, num_pixels, encoder_dim)
            decoder_hidden: Hidden state of decoder (batch_size, decoder_dim)
            
        Returns:
            attention weighted encoding, attention weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        
        # Calculate attention weights
        att = self.full_att(torch.tanh(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = F.softmax(att, dim=1)  # (batch_size, num_pixels)
        
        # Weight encoder outputs
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        
        return attention_weighted_encoding, alpha

class ShowAttendTell(BaseImageCaptioning):
    def __init__(self, vocab_size: int, embed_size: int = 256, hidden_size: int = 512, attention_dim: int = 256):
        super().__init__(vocab_size, embed_size, hidden_size)
        
        # Override encoder setup for spatial features
        self.fc = nn.Identity()  # Remove the flattening FC layer
        
        # Attention mechanism
        self.attention = Attention(
            encoder_dim=512,  # VGG16 feature channels
            decoder_dim=hidden_size,
            attention_dim=attention_dim
        )
        
        # Decoder LSTM
        self.decode_step = nn.LSTMCell(embed_size + 512, hidden_size)  # 512 is encoder feature dim
        self.init_h = nn.Linear(512, hidden_size)  # Initialize hidden state
        self.init_c = nn.Linear(512, hidden_size)  # Initialize cell state
        self.f_beta = nn.Linear(hidden_size, 512)  # Attention gating
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
        
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Extract spatial image features using CNN encoder."""
        features = self.encoder(images)  # (batch_size, 512, 7, 7)
        features = features.permute(0, 2, 3, 1)  # (batch_size, 7, 7, 512)
        features = features.view(features.size(0), -1, 512)  # (batch_size, 49, 512)
        return features
    
    def init_hidden_states(self, encoder_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden and cell states."""
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c
    
    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """Forward pass for training."""
        batch_size = images.size(0)
        
        # Extract image features
        encoder_out = self.encode_images(images)  # (batch_size, 49, 512)
        
        # Initialize LSTM states
        h, c = self.init_hidden_states(encoder_out)
        
        # Exclude last token (<end>) from captions
        decode_lengths = (captions != 0).sum(dim=1) - 1
        max_decode_lengths = decode_lengths.max()
        
        # Initialize tensors to store predictions and attention weights
        predictions = torch.zeros(batch_size, max_decode_lengths, self.vocab_size).to(images.device)
        alphas = torch.zeros(batch_size, max_decode_lengths, 49).to(images.device)
        
        # Embed captions
        embeddings = self.embed(captions)
        
        # Generate word by word
        for t in range(max_decode_lengths):
            # Attention weighted encoding
            attention_weighted_encoding, alpha = self.attention(encoder_out, h)
            gate = torch.sigmoid(self.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding
            
            # LSTM step
            h, c = self.decode_step(
                torch.cat([embeddings[:, t, :], attention_weighted_encoding], dim=1),
                (h, c)
            )
            
            # Generate prediction
            preds = self.fc(self.dropout(h))
            predictions[:, t] = preds
            alphas[:, t] = alpha
        
        # Calculate loss
        targets = captions
        
        # Cross entropy loss
        loss = nn.CrossEntropyLoss()(
            predictions.reshape(-1, self.vocab_size),
            targets[:, :max_decode_lengths].reshape(-1)
        )
        
        # Add doubly stochastic attention regularization
        loss += 1. * ((1. - alphas.sum(dim=1)) ** 2).mean()
        
        return loss
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, List[List[str]]], batch_idx: int) -> Dict:
        images, captions, original_captions = batch
        loss = self.forward(images, captions)
        
        # Log training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # In overfit mode, log the first batch images, captions and attention maps
        trainer = self.trainer
        is_overfit = trainer.overfit_batches > 0
        
        if is_overfit and batch_idx == 0:
            generated_captions, attention_maps = self.generate_caption(images, return_attention=True)
            self.log_images_and_captions(images, generated_captions, original_captions, attention_maps, prefix="train")
        
        return {'loss': loss}
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, List[List[str]]], batch_idx: int) -> Dict:
        images, captions, original_captions = batch
        loss = self.forward(images, captions)
        
        # Generate captions and get attention maps
        generated_captions, attention_maps = self.generate_caption(images, return_attention=True)
        
        # Log images and captions with attention maps
        # In overfit mode, log the first batch every time
        # In normal mode, log every 100 batches
        trainer = self.trainer
        is_overfit = trainer.overfit_batches > 0
        
        if (is_overfit and batch_idx == 0) or (not is_overfit and batch_idx % 100 == 0):
            self.log_images_and_captions(images, generated_captions, original_captions, attention_maps, prefix="val")
        
        # Store outputs for epoch end
        self.validation_step_outputs.append({
            'val_loss': loss,
            'generated_captions': generated_captions,
            'target_captions': captions
        })
        
        return {'val_loss': loss}
        
    def generate_caption(self, image: torch.Tensor, max_length: int = 20, return_attention: bool = False) -> Union[List[List[int]], Tuple[List[List[int]], torch.Tensor]]:
        """Generate captions for given images."""
        batch_size = image.size(0)
        captions = []
        all_attention_weights = []
        
        # Extract image features
        encoder_out = self.encode_images(image)  # (batch_size, 49, 512)
        
        for i in range(batch_size):
            # Initialize caption generation
            h, c = self.init_hidden_states(encoder_out[i:i+1])
            caption = []
            attention_weights = []
            
            # First input is <start> token embedding
            word = torch.tensor([1]).to(image.device)  # <start> token
            
            for _ in range(max_length):
                # Embed current word
                embeddings = self.embed(word)
                
                # Attention weighted encoding
                attention_weighted_encoding, alpha = self.attention(encoder_out[i:i+1], h)
                attention_weights.append(alpha)
                
                gate = torch.sigmoid(self.f_beta(h))
                attention_weighted_encoding = gate * attention_weighted_encoding
                
                # LSTM step
                h, c = self.decode_step(
                    torch.cat([embeddings, attention_weighted_encoding], dim=1),
                    (h, c)
                )
                
                # Generate prediction
                scores = self.fc(h)
                word = scores.argmax(1)
                
                # Add predicted word to caption
                caption.append(word.item())
                
                # Break if <end> token is predicted
                if word.item() == 2:  # <end> token
                    break
            
            captions.append(caption)
            if return_attention:
                all_attention_weights.append(torch.stack(attention_weights))
        
        if return_attention:
            # Pad attention weights to max length
            max_len = max(weights.size(0) for weights in all_attention_weights)
            padded_weights = []
            for weights in all_attention_weights:
                curr_len = weights.size(0)
                if curr_len < max_len:
                    padding = torch.zeros((max_len - curr_len,) + weights.shape[1:], device=weights.device)
                    padded_weights.append(torch.cat([weights, padding]))
                else:
                    padded_weights.append(weights)
            return captions, torch.stack(padded_weights)
        return captions 