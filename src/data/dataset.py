import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from typing import Dict, List, Tuple
from torch.nn.utils.rnn import pad_sequence

def flickr8k_collate_fn(batch):
    """Collate function to handle variable length sequences.
    
    Args:
        batch: List of tuples (image, caption)
        
    Returns:
        images: Tensor of shape (batch_size, 3, 224, 224)
        captions: Padded tensor of shape (batch_size, max_length)
    """
    # Separate images and captions
    images, captions = zip(*batch)
    
    # Stack images
    images = torch.stack(images, 0)
    
    # Pad captions
    captions = pad_sequence(captions, batch_first=True, padding_value=0)
    
    return images, captions

class Flickr8kDataset(Dataset):
    def __init__(self, 
                 image_dir: str,
                 captions_file: str,
                 transform=None,
                 split: str = 'train'):
        """
        Args:
            image_dir (str): Directory with all the images
            captions_file (str): Path to the captions file
            transform: Optional transform to be applied on images
            split (str): train/val/test split
        """
        self.image_dir = image_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Load captions
        self.captions_data = pd.read_csv(captions_file)
        
        # Filter by split
        split_data = self.captions_data[self.captions_data['split'] == split]
        
        # Group captions by image
        grouped_data = split_data.groupby('image')
        self.image_filenames = list(grouped_data.groups.keys())
        self.image_to_captions = {img: group['caption'].tolist() for img, group in grouped_data}
        
        # For iteration, use first caption of each image
        self.captions = [self.image_to_captions[img][0] for img in self.image_filenames]
        
        # Build vocabulary
        self.vocab = self._build_vocabulary()
        
    def _build_vocabulary(self) -> Dict[str, int]:
        """Build a simple vocabulary wrapper."""
        words = set()
        for captions in self.image_to_captions.values():
            for caption in captions:
                words.update(caption.lower().split())
        
        vocab = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
        for i, word in enumerate(sorted(words)):
            vocab[word] = i + 4
        return vocab
    
    def _process_caption(self, caption: str) -> torch.Tensor:
        """Convert caption string to tensor of indices."""
        tokens = ['<start>'] + caption.lower().split() + ['<end>']
        token_ids = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        return torch.tensor(token_ids, dtype=torch.long)
    
    def __len__(self) -> int:
        return len(self.image_filenames)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Process caption (use first caption for training)
        caption = self._process_caption(self.captions[idx])
        
        return image, caption
    
    def get_all_captions(self, image_filename: str) -> List[str]:
        """Get all 5 captions for a given image filename."""
        return self.image_to_captions.get(image_filename, [])
    
    @property
    def vocab_size(self) -> int:
        return len(self.vocab) 