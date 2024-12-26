import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from .base_model import BaseImageCaptioning

class ShowAndTell(BaseImageCaptioning):
    def __init__(self, vocab_size: int, embed_size: int = 256, hidden_size: int = 512):
        super().__init__(vocab_size, embed_size, hidden_size)
        
        # LSTM decoder
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """Forward pass for training."""
        batch_size = images.size(0)
        
        # Extract image features
        features = self.encode_images(images)  # (batch_size, embed_size)
        
        # Embed captions (exclude last token)
        embeddings = self.embed(captions[:, :-1])  # (batch_size, seq_len-1, embed_size)
        
        # Concatenate image features with caption embeddings
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)  # (batch_size, seq_len, embed_size)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(inputs)  # (batch_size, seq_len, hidden_size)
        outputs = self.linear(lstm_out)  # (batch_size, seq_len, vocab_size)
        
        # Calculate cross entropy loss
        # Reshape outputs to (batch_size * seq_len, vocab_size)
        outputs = outputs.reshape(-1, self.vocab_size)
        # Use targets starting from the second token (exclude <start>)
        targets = captions[:, 1:].reshape(-1)
        
        loss = nn.CrossEntropyLoss(ignore_index=0)(outputs, targets)
        
        return loss
    
    def generate_caption(self, image: torch.Tensor, max_length: int = 20) -> List[List[int]]:
        """Generate captions for given images."""
        batch_size = image.size(0)
        captions = []
        
        for i in range(batch_size):
            # Initialize caption generation
            feature = self.encode_images(image[i:i+1])
            caption = []
            hidden = None
            
            # First input is the image feature
            inputs = feature.unsqueeze(1)
            
            # Generate caption word by word
            for _ in range(max_length):
                lstm_out, hidden = self.lstm(inputs, hidden)
                outputs = self.linear(lstm_out.squeeze(1))
                predicted = outputs.argmax(1)
                
                # Add predicted word to caption
                caption.append(predicted.item())
                
                # Break if <end> token is predicted
                if predicted.item() == 2:  # <end> token
                    break
                
                # Prepare input for next iteration
                inputs = self.embed(predicted).unsqueeze(1)
            
            captions.append(caption)
        
        return captions 