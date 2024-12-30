import os
import argparse
import torch
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import corpus_bleu
import wandb
from tqdm import tqdm
import numpy as np

from src.data.dataset import Flickr8kDataset, flickr8k_collate_fn
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
    
    # For top-k accuracy
    total_words = 0
    top1_hits = 0
    top5_hits = 0
    
    # Initialize wandb for logging
    wandb.init(project="image-captioning-comparison", name=f"eval_{model.__class__.__name__}")
    
    with torch.no_grad():
        for images, captions, original_captions in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            captions = captions.to(device)  # Move captions to device for accuracy calculation
            
            # Generate captions (and attention maps for Show, Attend and Tell)
            if isinstance(model, ShowAttendTell):
                generated_captions, attention_maps = model.generate_caption(images, return_attention=True)
                
                # Convert images for attention grid visualization
                images_np = [model.inverse_transform(img).cpu().numpy().transpose(1, 2, 0) for img in images]
                images_np = [np.clip(img, 0, 1) for img in images_np]
                
                # Create attention grids for each image
                attention_grids = []
                for idx, (image, gen_caption, attn_maps) in enumerate(zip(images_np, generated_captions, attention_maps)):
                    attention_grid = model.create_attention_grid(image, gen_caption, attn_maps)
                    if attention_grid is not None:
                        attention_grids.append(wandb.Image(attention_grid, caption=f"Word-by-word attention for image {idx}"))
                
                # Log images with attention maps and grids
                model.log_images_and_captions(images, generated_captions, original_captions, attention_maps, prefix="test")
                if attention_grids:
                    wandb.log({"test_attention_grids": attention_grids})
            else:
                generated_captions = model.generate_caption(images)
                # Log images without attention maps
                model.log_images_and_captions(images, generated_captions, original_captions, prefix="test")
            
            # Convert generated captions and references to words
            for gen_caption, target_caption, orig_captions in zip(generated_captions, captions, original_captions):
                # Process generated caption for BLEU
                hypothesis = convert_ids_to_words(gen_caption, data_loader.dataset.vocab)
                all_hypotheses.append(hypothesis)
                
                # Process all 5 reference captions for BLEU
                references = []
                for ref_caption in orig_captions:
                    # Split the caption string into words
                    reference = ref_caption.lower().split()
                    references.append(reference)
                all_references.append(references)  # Add all 5 references for this hypothesis
                
                # Calculate top-k accuracy
                # Compare each generated word with target word (excluding <start> and special tokens)
                for gen_word, target_word in zip(gen_caption[:-1], target_caption[1:]):  # Shift target by 1
                    if target_word == 0:  # Skip padding
                        continue
                    total_words += 1
                    if gen_word == target_word:  # Top-1 hit
                        top1_hits += 1
                    # For top-5, we need to run the model's forward pass
                    with torch.set_grad_enabled(False):
                        if isinstance(model, ShowAttendTell):
                            # Get encoder features
                            encoder_out = model.encode_images(images[0:1])  # Use first image
                            # Initialize states
                            h, c = model.init_hidden_states(encoder_out)
                            # Get attention and prediction
                            attention_weighted_encoding, _ = model.attention(encoder_out, h)
                            gate = torch.sigmoid(model.f_beta(h))
                            attention_weighted_encoding = gate * attention_weighted_encoding
                            # Get word embedding
                            word_embed = model.embed(torch.tensor([gen_word], device=device))
                            # LSTM step
                            h, c = model.decode_step(
                                torch.cat([word_embed, attention_weighted_encoding], dim=1),
                                (h, c)
                            )
                            # Get prediction
                            scores = model.fc(h)
                        else:
                            # For Show and Tell, simpler forward pass
                            feature = model.encode_images(images[0:1])
                            hidden = None
                            lstm_out, hidden = model.lstm(feature.unsqueeze(1), hidden)
                            scores = model.linear(lstm_out.squeeze(1))
                        
                        # Check if target is in top 5
                        if target_word in torch.topk(F.softmax(scores, dim=1), k=5)[1]:
                            top5_hits += 1
    
    # Calculate BLEU scores
    bleu1 = corpus_bleu(all_references, all_hypotheses, weights=(1.0, 0, 0, 0))
    bleu2 = corpus_bleu(all_references, all_hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(all_references, all_hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(all_references, all_hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    
    # Calculate top-k accuracy
    top1_acc = (top1_hits / total_words) * 100 if total_words > 0 else 0
    top5_acc = (top5_hits / total_words) * 100 if total_words > 0 else 0
    
    # Log metrics to wandb
    wandb.log({
        'test_bleu1': bleu1 * 100,
        'test_bleu2': bleu2 * 100,
        'test_bleu3': bleu3 * 100,
        'test_bleu4': bleu4 * 100,
        'test_top1_acc': top1_acc,
        'test_top5_acc': top5_acc
    })
    
    return {
        'bleu1': bleu1 * 100,
        'bleu2': bleu2 * 100,
        'bleu3': bleu3 * 100,
        'bleu4': bleu4 * 100,
        'top1_acc': top1_acc,
        'top5_acc': top5_acc
    }

def main(args):
    # Data loading
    test_dataset = Flickr8kDataset(
        image_dir=os.path.join(DATA_DIR, 'Images'),
        captions_file=os.path.join(DATA_DIR, 'captions.txt'),
        split='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=flickr8k_collate_fn
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
    model.to(device)
    
    # Set vocabulary
    model.set_vocabulary(test_dataset.vocab)
    
    # Evaluate
    metrics = evaluate(model, test_loader, device)
    
    # Print results
    print("\nTest Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")

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
    
    # Data loading arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    main(args) 