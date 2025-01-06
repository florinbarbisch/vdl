import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import json
from typing import Dict, List, Tuple
from torch.nn.utils.rnn import pad_sequence
import numpy as np

def flickr8k_collate_fn(batch):
    """Collate function to handle variable length sequences.
    
    Args:
        batch: List of tuples (image, caption, all_captions)
        
    Returns:
        images: Tensor of shape (batch_size, 3, 224, 224)
        captions: Padded tensor of shape (batch_size, max_length)
        all_captions: List of lists containing all 5 captions for each image
    """
    # Separate images, captions and all_captions
    images, captions, all_captions = zip(*batch)
    
    # Stack images
    images = torch.stack(images, 0)
    
    # Pad captions
    captions = pad_sequence(captions, batch_first=True, padding_value=0)
    
    return images, captions, list(all_captions)

class Flickr8kDataset(Dataset):
    def __init__(self, 
                 image_dir: str,
                 captions_file: str,
                 transform=None,
                 split: str = 'train',
                 eval_fold: int = None):
        """
        Args:
            image_dir (str): Directory with all the images
            captions_file (str): Path to the captions file
            transform: Optional transform to be applied on images
            split (str): train/val/test split
            eval_fold (int): Which fold to use for evaluation (0-4). 
                           If specified, this fold will be split into val/test,
                           and other folds will be used for training.
                           Also determines which fold-specific vocabulary to use.
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
        
        if eval_fold is not None:
            # Cross-validation mode
            if eval_fold < 0 or eval_fold > 4:
                raise ValueError("eval_fold must be between 0 and 4")
            
            # Load fold-specific vocabulary
            vocab_file = os.path.join(os.path.dirname(captions_file), "vocab", f"vocab_fold{eval_fold}.json")
            if not os.path.exists(vocab_file):
                raise FileNotFoundError(
                    f"Fold-specific vocabulary file not found at {vocab_file}. "
                    "Please run prepare_dataset.py first to create the vocabularies."
                )
            with open(vocab_file, 'r') as f:
                self.vocab = json.load(f)
            
            if split == 'train':
                # Use all folds except eval_fold for training
                split_data = self.captions_data[self.captions_data['fold'] != eval_fold]
            else:
                # Get eval fold data
                eval_data = self.captions_data[self.captions_data['fold'] == eval_fold]
                
                # Get unique images in eval fold
                eval_images = eval_data['image'].unique()
                
                # Split eval fold images into val (50%) and test (50%) deterministically
                rng = np.random.RandomState(42 + eval_fold)  # Unique seed for each fold
                eval_images = np.array(eval_images)
                rng.shuffle(eval_images)
                split_idx = len(eval_images) // 2
                
                val_images = eval_images[:split_idx]
                test_images = eval_images[split_idx:]
                
                # Filter data based on split
                if split == 'val':
                    split_data = eval_data[eval_data['image'].isin(val_images)]
                else:  # test
                    split_data = eval_data[eval_data['image'].isin(test_images)]
        else:
            # Use original split column (for backward compatibility)
            split_data = self.captions_data[self.captions_data['split'] == split]
            
            # Load full vocabulary (backward compatibility)
            vocab_file = os.path.join(os.path.dirname(captions_file), "vocab.json")
            if not os.path.exists(vocab_file):
                raise FileNotFoundError(
                    f"Vocabulary file not found at {vocab_file}. "
                    "Please run prepare_dataset.py first to create the vocabulary."
                )
            with open(vocab_file, 'r') as f:
                self.vocab = json.load(f)
        
        # Group captions by image
        grouped_data = split_data.groupby('image')
        self.image_filenames = list(grouped_data.groups.keys())
        self.image_to_captions = {img: group['caption'].tolist() for img, group in grouped_data}
        
        # Create a list of (image_filename, caption) pairs for iteration
        self.samples = []
        for img in self.image_filenames:
            for caption in self.image_to_captions[img]:
                self.samples.append((img, caption))
        
    def _process_caption(self, caption: str) -> torch.Tensor:
        """Convert caption string to tensor of indices."""
        tokens = ['<start>'] + caption.lower().split() + ['<end>']
        token_ids = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        return torch.tensor(token_ids, dtype=torch.long)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        # Get image filename and caption for this index
        image_filename, caption = self.samples[idx]
        
        # Load image
        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Get all captions for this image
        all_captions = self.image_to_captions[image_filename]
        
        # Process caption
        caption_tensor = self._process_caption(caption)
        
        return image, caption_tensor, all_captions
    
    @property
    def vocab_size(self) -> int:
        return len(self.vocab) 