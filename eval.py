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

def evaluate(model, data_loader, device, beam_size: int = 3):
    model.eval()
    all_references = []
    all_hypotheses_greedy = []
    all_hypotheses_beam = []
    
    # Initialize wandb for logging
    wandb.init(project="image-captioning-comparison", name=f"eval_{model.__class__.__name__}")
    
    with torch.no_grad():
        for batch_idx, (images, captions, original_captions) in enumerate(tqdm(data_loader, desc="Evaluating")):
            images = images.to(device)
            
            # Generate captions using both greedy and beam search
            if isinstance(model, ShowAttendTell):
                # Greedy search with attention
                greedy_captions, greedy_attention = model.generate_caption(images, return_attention=True)
                # Beam search with attention
                beam_results, beam_attention = model.generate_caption(images, beam_size=beam_size, return_attention=True)
                beam_captions = [beam[0][0] for beam in beam_results]  # Take the top beam for each image
                
                # Only create and log attention grids every 8th batch for 8 images
                if batch_idx % 8 == 0:
                    # Convert images for visualization
                    images_np = [model.inverse_transform(img).cpu().numpy().transpose(1, 2, 0) for img in images[:8]]
                    images_np = [np.clip(img, 0, 1) for img in images_np]
                    
                    # Create attention grids for both methods
                    attention_grids = []
                    for idx, (image, greedy_cap, beam_cap, greedy_attn, beam_attn) in enumerate(zip(
                        images_np, greedy_captions, beam_captions, 
                        greedy_attention, beam_attention
                    )):
                        if idx % 8 != 0:  # Process only every 8th image
                            continue
                        # Create attention grids
                        greedy_grid = model.create_attention_grid(image, greedy_cap, greedy_attn)
                        beam_grid = model.create_attention_grid(image, beam_cap, beam_attn)
                        
                        if greedy_grid is not None and beam_grid is not None:
                            # Combine grids side by side
                            # Make both grids the same height by padding the shorter one
                            max_height = max(greedy_grid.shape[0], beam_grid.shape[0])
                            if greedy_grid.shape[0] < max_height:
                                pad_height = max_height - greedy_grid.shape[0]
                                greedy_grid = np.vstack([greedy_grid, np.ones((pad_height, greedy_grid.shape[1], *greedy_grid.shape[2:]))])
                            elif beam_grid.shape[0] < max_height:
                                pad_height = max_height - beam_grid.shape[0]
                                beam_grid = np.vstack([beam_grid, np.ones((pad_height, beam_grid.shape[1], *beam_grid.shape[2:]))])
                            combined_grid = np.hstack([greedy_grid, beam_grid])
                            attention_grids.append(wandb.Image(
                                combined_grid,
                                caption=f"Left: Greedy Search Attention | Right: Beam Search Attention (image {idx})"
                            ))
                    
                    # Log images with attention maps and grids
                    model.log_images_and_captions(
                        images[:8], greedy_captions[:8], original_captions[:8], 
                        greedy_attention[:8], prefix="test_greedy"
                    )
                    model.log_images_and_captions(
                        images[:8], beam_captions[:8], original_captions[:8], 
                        beam_attention[:8], prefix="test_beam"
                    )
                    if attention_grids:
                        wandb.log({"test_attention_comparison": attention_grids})
            else:
                # For Show and Tell model (no attention)
                greedy_captions = model.generate_caption(images)
                beam_results = model.generate_caption(images, beam_size=beam_size)
                beam_captions = [beam[0][0] for beam in beam_results]  # Take the top beam for each image
                
                # Log images and captions
                if batch_idx % 8 == 0:
                    model.log_images_and_captions(images[:8], greedy_captions[:8], original_captions[:8], prefix="test_greedy")
                    model.log_images_and_captions(images[:8], beam_captions[:8], original_captions[:8], prefix="test_beam")
            
            # Convert generated captions and references to words
            for greedy_cap, beam_cap, orig_captions in zip(greedy_captions, beam_captions, original_captions):
                # Process greedy caption
                greedy_hypothesis = convert_ids_to_words(greedy_cap, data_loader.dataset.vocab)
                all_hypotheses_greedy.append(greedy_hypothesis)
                
                # Process beam search caption
                beam_hypothesis = convert_ids_to_words(beam_cap, data_loader.dataset.vocab)
                all_hypotheses_beam.append(beam_hypothesis)
                
                # Process all 5 reference captions (only need to do this once)
                if len(all_references) < len(all_hypotheses_greedy):
                    references = []
                    for ref_caption in orig_captions:
                        # Split the caption string into words
                        reference = ref_caption.lower().split()
                        references.append(reference)
                    all_references.append(references)
    
    # Calculate BLEU scores for both methods
    metrics = {}
    
    # Greedy search metrics
    metrics['greedy_bleu1'] = corpus_bleu(all_references, all_hypotheses_greedy, weights=(1.0, 0, 0, 0)) * 100
    metrics['greedy_bleu2'] = corpus_bleu(all_references, all_hypotheses_greedy, weights=(0.5, 0.5, 0, 0)) * 100
    metrics['greedy_bleu3'] = corpus_bleu(all_references, all_hypotheses_greedy, weights=(0.33, 0.33, 0.33, 0)) * 100
    metrics['greedy_bleu4'] = corpus_bleu(all_references, all_hypotheses_greedy, weights=(0.25, 0.25, 0.25, 0.25)) * 100
    
    # Beam search metrics
    metrics['beam_bleu1'] = corpus_bleu(all_references, all_hypotheses_beam, weights=(1.0, 0, 0, 0)) * 100
    metrics['beam_bleu2'] = corpus_bleu(all_references, all_hypotheses_beam, weights=(0.5, 0.5, 0, 0)) * 100
    metrics['beam_bleu3'] = corpus_bleu(all_references, all_hypotheses_beam, weights=(0.33, 0.33, 0.33, 0)) * 100
    metrics['beam_bleu4'] = corpus_bleu(all_references, all_hypotheses_beam, weights=(0.25, 0.25, 0.25, 0.25)) * 100
    
    # Log BLEU scores to wandb
    wandb.log(metrics)
    
    return metrics

def download_model_from_wandb(artifact_path: str) -> str:
    """Download model checkpoint from wandb artifacts.
    
    Args:
        artifact_path: Path to the artifact in format 'entity/project/artifact_name:version'
                      or just 'artifact_name:version' if already in a wandb run
    
    Returns:
        Local path to the downloaded checkpoint file
    """
    artifact = wandb.use_artifact(artifact_path)
    artifact_dir = artifact.download()
    return os.path.join(artifact_dir, "model.ckpt")

def main(args):
    # Initialize wandb if using wandb artifacts
    if args.wandb_artifact:
        if args.wandb_project:
            wandb.init(project=args.wandb_project)
        checkpoint_path = download_model_from_wandb(args.wandb_artifact)
    else:
        checkpoint_path = args.checkpoint_path
    
    # Data loading
    test_dataset = Flickr8kDataset(
        image_dir=os.path.join(DATA_DIR, 'Images'),
        captions_file=os.path.join(DATA_DIR, 'captions.txt'),
        split='test',
        eval_fold=args.eval_fold
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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    
    # Set vocabulary
    model.set_vocabulary(test_dataset.vocab)
    
    # Initialize wandb run for logging if not already initialized
    if not wandb.run:
        run_name = f"eval_{args.model}"
        if args.eval_fold is not None:
            run_name += f"_fold{args.eval_fold}"
        wandb.init(project="image-captioning-comparison", name=run_name)
    
    # Evaluate
    metrics = evaluate(model, test_loader, device, args.beam_size)
    
    # Print results
    print("\nTest Results:")
    if args.eval_fold is not None:
        print(f"\nFold {args.eval_fold}:")
    print("\nGreedy Search:")
    for metric, value in metrics.items():
        if metric.startswith('greedy_'):
            print(f"{metric[7:]}: {value:.2f}")
    print("\nBeam Search:")
    for metric, value in metrics.items():
        if metric.startswith('beam_'):
            print(f"{metric[5:]}: {value:.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate image captioning models')
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True,
                        choices=['show_tell', 'show_attend_tell'],
                        help='Model architecture to use')
    parser.add_argument('--checkpoint_path', type=str,
                        help='Path to local model checkpoint')
    parser.add_argument('--wandb_artifact', type=str,
                        help='Path to wandb artifact (e.g., "model-show_tell_20240101_120000:latest")')
    parser.add_argument('--wandb_project', type=str,
                        help='Wandb project name (required if using wandb_artifact)')
    parser.add_argument('--embed_size', type=int, default=256,
                        help='Word embedding size')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='LSTM hidden size')
    parser.add_argument('--attention_dim', type=int, default=256,
                        help='Attention layer dimension (for Show, Attend and Tell)')
    parser.add_argument('--beam_size', type=int, default=3,
                        help='Beam size for beam search')
    parser.add_argument('--eval_fold', type=int, choices=[0, 1, 2, 3, 4],
                        help='Which fold to use for evaluation in cross-validation')
    
    # Data loading arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.checkpoint_path and not args.wandb_artifact:
        parser.error("Either --checkpoint_path or --wandb_artifact must be provided")
    if args.wandb_artifact and not args.wandb_project:
        parser.error("--wandb_project is required when using --wandb_artifact")
    
    main(args) 