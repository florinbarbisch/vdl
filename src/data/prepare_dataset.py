import os
import pandas as pd
import kaggle
import json
from sklearn.model_selection import train_test_split

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
    """Prepare captions.txt by adding train/val/test split."""
    print("Preparing captions file...")
    
    # Read captions
    df = pd.read_csv(captions_file)
    
    # Get unique image names
    unique_images = df['image'].unique()
    
    # Split images into train (80%), validation (10%), and test (10%)
    train_images, temp_images = train_test_split(unique_images, test_size=0.2, random_state=42)
    val_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)
    
    # Create split column
    df['split'] = 'train'  # default value
    df.loc[df['image'].isin(val_images), 'split'] = 'val'
    df.loc[df['image'].isin(test_images), 'split'] = 'test'
    
    # Build vocabulary using only training captions
    train_df = df[df['split'] == 'train']
    vocab = build_vocabulary(train_df)
    
    # Save vocabulary
    vocab_file = os.path.join(os.path.dirname(captions_file), "vocab.json")
    with open(vocab_file, 'w') as f:
        json.dump(vocab, f)
    print(f"Vocabulary saved to: {vocab_file}")
    
    # Save updated captions file
    df.to_csv(captions_file, index=False)
    print("Captions file updated with splits")
    
    # Print split statistics
    print("\nDataset split statistics:")
    print(df['split'].value_counts())

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