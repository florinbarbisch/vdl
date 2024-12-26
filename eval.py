import os
import argparse
import torch
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import corpus_bleu
import wandb

from src.data.dataset import Flickr8kDataset
from src.models.show_and_tell import ShowAndTell
from src.models.show_attend_tell import ShowAttendTell

# Fixed data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def convert_ids_to_words(caption_ids: list, vocab: dict) -> list:
    """Convert caption indices to words."""
    # Create reverse vocabulary (id to word mapping)
    id_to_word = {v: k for k, v in vocab.items()}
    
    # Convert ids to words, excluding special tokens
    words = []
    for idx in caption_ids:
        if idx == 2:  # <end> token
            break
        if idx >= 4:  # Skip special tokens
            words.append(id_to_word[idx])
    
    return words

def evaluate(model, data_loader, device):
    model.eval()
    all_references = []
    all_hypotheses = []
    
    with torch.no_grad():
        for images, captions in data_loader:
            images = images.to(device)
            
            # Generate captions
            generated_captions = model.generate_caption(images)
            
            # Convert generated captions and references to words
            for gen_caption, ref_caption in zip(generated_captions, captions):
                # Process generated caption
                hypothesis = convert_ids_to_words(gen_caption, data_loader.dataset.vocab)
                all_hypotheses.append(hypothesis)
                
                # Process reference caption
                reference = convert_ids_to_words(ref_caption.tolist(), data_loader.dataset.vocab)
                all_references.append([reference])  # BLEU expects list of references for each hypothesis
    
    # Calculate BLEU scores
    bleu1 = corpus_bleu(all_references, all_hypotheses, weights=(1.0, 0, 0, 0))
    bleu2 = corpus_bleu(all_references, all_hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(all_references, all_hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(all_references, all_hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    
    return {
        'bleu1': bleu1 * 100,
        'bleu2': bleu2 * 100,
        'bleu3': bleu3 * 100,
        'bleu4': bleu4 * 100
    }

def main(args):
    # Initialize wandb
    wandb.init(project="image-captioning-comparison", name=f"{args.model}-evaluation")
    
    # Load test dataset
    test_dataset = Flickr8kDataset(
        image_dir=os.path.join(DATA_DIR, 'images'),
        captions_file=os.path.join(DATA_DIR, 'captions.txt'),
        split='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    if args.model == 'show_tell':
        model = ShowAndTell(
            vocab_size=len(test_dataset.vocab),
            embed_size=args.embed_size,
            hidden_size=args.hidden_size
        )
    else:
        model = ShowAttendTell(
            vocab_size=len(test_dataset.vocab),
            embed_size=args.embed_size,
            hidden_size=args.hidden_size,
            attention_dim=args.attention_dim
        )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    
    # Evaluate
    metrics = evaluate(model, test_loader, device)
    
    # Log results
    print(f"\nResults for {args.model}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")
        wandb.log({metric: value})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate image captioning models')
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True,
                        choices=['show_tell', 'show_attend_tell'],
                        help='Model architecture to use')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--embed_size', type=int, default=256,
                        help='Word embedding size')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='LSTM hidden size')
    parser.add_argument('--attention_dim', type=int, default=256,
                        help='Attention layer dimension (for Show, Attend and Tell)')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    main(args) 