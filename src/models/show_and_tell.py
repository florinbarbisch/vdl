import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union
from .base_model import BaseImageCaptioning

class ShowAndTell(BaseImageCaptioning):
    def __init__(self, vocab_size: int, embed_size: int = 256, hidden_size: int = 512):
        super().__init__(vocab_size, embed_size, hidden_size)
        
        # Replace LSTM with LSTMCell
        self.decode_step = nn.LSTMCell(embed_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
        
        # Keep initialization layers
        self.init_h = nn.Linear(embed_size, hidden_size)
        self.init_c = nn.Linear(embed_size, hidden_size)
    
    def init_hidden_states(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden and cell states."""
        h = self.init_h(features)
        c = self.init_c(features)
        return h.unsqueeze(0), c.unsqueeze(0)  # Add sequence dimension for LSTM
        
    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """Forward pass for training."""
        batch_size = images.size(0)
        
        # Extract image features
        features = self.encode_images(images)
        
        # Initialize LSTM states
        h, c = self.init_hidden_states(features)
        
        # Exclude last token from captions
        decode_lengths = (captions != 0).sum(dim=1) - 1
        max_decode_lengths = decode_lengths.max()
        
        # Initialize predictions tensor
        predictions = torch.zeros(batch_size, max_decode_lengths, self.vocab_size).to(images.device)
        
        # Embed captions
        embeddings = self.embed(captions)
        
        # Generate word by word
        for t in range(max_decode_lengths):
            h, c = self.decode_step(embeddings[:, t, :], (h, c))
            predictions[:, t] = self.fc(self.dropout(h))
        
        # Calculate loss
        targets = captions[:, 1:max_decode_lengths + 1]
        loss = nn.CrossEntropyLoss(ignore_index=0)(
            predictions.reshape(-1, self.vocab_size),
            targets.reshape(-1)
        )
        
        return loss
    
    def generate_caption(self, image: torch.Tensor, max_length: int = 20, beam_size: int = None) -> Union[List[List[int]], List[List[Tuple[List[int], float]]]]:
        """Generate captions for given images using either greedy search or beam search."""
        if beam_size is None or beam_size == 1:
            return self._generate_caption_greedy(image, max_length)
        else:
            return self._generate_caption_beam(image, max_length, beam_size)
    
    def _generate_caption_greedy(self, image: torch.Tensor, max_length: int = 20) -> List[List[int]]:
        """Generate captions using greedy search."""
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
    
    def _generate_caption_beam(self, image: torch.Tensor, max_length: int = 20, beam_size: int = 3) -> List[List[Tuple[List[int], float]]]:
        """Generate captions using beam search."""
        batch_size = image.size(0)
        all_captions = []
        
        for i in range(batch_size):
            # Initialize caption generation
            feature = self.encode_images(image[i:i+1])
            
            # First input is the image feature
            inputs = feature.unsqueeze(1)
            
            # Get initial hidden state
            lstm_out, hidden = self.lstm(inputs)
            outputs = self.linear(lstm_out.squeeze(1))
            
            # Initialize beam with top-k most likely first words
            log_probs = F.log_softmax(outputs, dim=1)
            topk_log_probs, topk_words = log_probs.topk(beam_size, 1)
            
            # Initialize beams: (sequence, score, hidden_state, cell_state)
            beams = [(
                [word.item()],
                score.item(),
                hidden[0].squeeze(0),
                hidden[1].squeeze(0)
            ) for word, score in zip(topk_words[0], topk_log_probs[0])]
            
            # Expand beams
            for _ in range(max_length - 1):
                candidates = []
                
                # Expand each beam
                for sequence, score, h, c in beams:
                    if sequence[-1] == 2:  # if last token is <end>
                        candidates.append((sequence, score, h, c))
                        continue
                        
                    # Get predictions for next word
                    word_input = torch.tensor([sequence[-1]], device=image.device)
                    inputs = self.embed(word_input).unsqueeze(1)
                    lstm_out, (h_new, c_new) = self.lstm(inputs, (h.unsqueeze(0), c.unsqueeze(0)))
                    outputs = self.linear(lstm_out.squeeze(1))
                    log_probs = F.log_softmax(outputs, dim=1)
                    
                    # Get top k words
                    topk_log_probs, topk_words = log_probs.topk(beam_size, 1)
                    
                    # Create new candidates
                    for word, word_score in zip(topk_words[0], topk_log_probs[0]):
                        word_item = word.item()
                        candidates.append((
                            sequence + [word_item],
                            score + word_score.item(),
                            h_new.squeeze(0),
                            c_new.squeeze(0)
                        ))
                
                # Select top k candidates
                beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
                
                # Early stopping if all beams end with <end>
                if all(beam[0][-1] == 2 for beam in beams):
                    break
            
            # Add final beam results (sequence and score) for this image
            all_captions.append([(beam[0], beam[1]) for beam in beams])
        
        return all_captions 