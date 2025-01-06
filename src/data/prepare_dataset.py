import os
import pandas as pd
import kaggle
import json
from sklearn.model_selection import train_test_split
import numpy as np

def download_dataset():
    """Download the Flickr8k dataset if not already present."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    captions_file = os.path.join(data_dir, "captions.txt")
    images_dir = os.path.join(data_dir, "Images")
    
    if not os.path.exists(captions_file) and (not os.path.exists(images_dir) or not any(f.endswith('.jpg') for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f)))):
        print("Downloading Flickr8k dataset...")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files("adityajn105/flickr8k", path=data_dir, unzip=True)
        print("Dataset downloaded to:", data_dir)
    return data_dir

def build_vocabulary(df: pd.DataFrame, min_freq: int = 3) -> dict:
    """Build vocabulary from captions with minimum frequency threshold."""
    print("Building vocabulary...")
    word_freq = {}
    
    # Count word frequencies across all captions
    for caption in df['caption']:
        for word in caption.lower().split():
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Create vocabulary with words above threshold
    vocab = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
    for word, freq in sorted(word_freq.items()):
        if freq >= min_freq:
            vocab[word] = len(vocab)
    
    print(f"Vocabulary size: {len(vocab)}")
    return vocab

def prepare_captions(captions_file: str):
    """Prepare captions.txt by adding fold assignments and creating fold-specific vocabularies."""
    print("Preparing captions file...")
    
    # Read captions
    df = pd.read_csv(captions_file)
    
    # Get unique image names
    unique_images = df['image'].unique()
    
    # Assign folds (0-4) to images using deterministic shuffle
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility
    shuffled_images = np.array(unique_images)
    rng.shuffle(shuffled_images)
    
    # Create fold assignments (0-4)
    fold_assignments = {img: i % 5 for i, img in enumerate(shuffled_images)}
    
    # Add fold column to dataframe
    df['fold'] = df['image'].map(fold_assignments)
    
    # Default split is train
    df['split'] = 'train'
    
    # Create vocabularies for each fold using only training data
    vocab_dir = os.path.join(os.path.dirname(captions_file), "vocab")
    os.makedirs(vocab_dir, exist_ok=True)
    
    print("\nCreating fold-specific vocabularies:")
    for fold in range(5):
        # Get training data for this fold (all data except current fold)
        train_df = df[df['fold'] != fold]
        
        # Build vocabulary using only training data
        vocab = build_vocabulary(train_df)
        
        # Save vocabulary for this fold
        vocab_file = os.path.join(vocab_dir, f"vocab_fold{fold}.json")
        with open(vocab_file, 'w') as f:
            json.dump(vocab, f)
        print(f"Fold {fold}: Vocabulary size = {len(vocab)}")
    
    # Also create a full vocabulary using all data (for backward compatibility)
    full_vocab = build_vocabulary(df)
    full_vocab_file = os.path.join(os.path.dirname(captions_file), "vocab.json")
    with open(full_vocab_file, 'w') as f:
        json.dump(full_vocab, f)
    print(f"\nFull vocabulary size (all data): {len(full_vocab)}")
    
    # Save updated captions file with fold assignments
    df.to_csv(captions_file, index=False)
    print("\nCaptions file updated with fold assignments")
    
    # Print fold statistics
    print("\nFold statistics:")
    fold_counts = df.groupby('fold')['image'].nunique()
    for fold, count in fold_counts.items():
        print(f"Fold {fold}: {count} unique images")

def main():
    # Download dataset
    dataset_path = download_dataset()
    
    # Prepare captions
    captions_file = os.path.join(dataset_path, "captions.txt")
    if os.path.exists(captions_file):
        prepare_captions(captions_file)
    else:
        print(f"Error: Captions file not found at {captions_file}")

if __name__ == "__main__":
    main()