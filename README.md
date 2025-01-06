# Image Captioning Models Comparison

This project implements and compares two image captioning architectures:
1. Show and Tell (Baseline)
2. Show, Attend and Tell (with Soft Attention)

## Project Structure
```
├── data/               # Data directory for Flickr8k dataset
├── src/
│   ├── models/        # Model implementations
│   ├── data/          # Data loading and processing
│   ├── utils/         # Utility functions
│   └── train.py       # Training script
└── eval.py            # Evaluation script
```

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Login to Kaggle:
   - Download your kaggle.json from your Kaggle account settings
   - Place it in `~/.kaggle/kaggle.json`

3. Prepare the dataset:
```bash
python src/data/prepare_dataset.py
```
This script will:
- Download the Flickr8k dataset using kaggle
- Add 5 fold split for cross-validation

## Training
Normal training:
```bash
python src/train.py --model [show_tell/show_attend_tell] --eval_fold 0
```

Debug modes:
```bash
# Overfitting test (uses 1% of training data)
python src/train.py --model [show_tell/show_attend_tell] --eval_fold 0 --debug overfit

# Quick pipeline test (runs 1 batch through train and val)
python src/train.py --model [show_tell/show_attend_tell] --eval_fold 0 --debug fast
```

The debug modes:
- `overfit`: 
  - Uses 1% of the training data
  - Runs for 100 epochs
  - Disables checkpointing and wandb logging
  - Useful for verifying model can overfit to a small dataset

- `fast`:
  - Runs only 1 batch through training and validation
  - Disables checkpointing and wandb logging
  - Useful for quickly verifying the training pipeline works

## Evaluation
```bash
python eval.py --model [show_tell/show_attend_tell] --eval_fold 0 --checkpoint_path [path_to_checkpoint]
```

Or from wandb:
```bash
python eval.py --model [show_tell/show_attend_tell] --eval_fold 0 --wandb_project [wandb_project_name] --wandb_artifact [artifact_name:version]
```

## Models
- Both models use a pre-trained VGGNet as the CNN encoder
- LSTM decoder for caption generation
- Show, Attend and Tell implements soft attention mechanism
- Models are implemented using PyTorch Lightning
- Training progress is logged using Weights & Biases (wandb)

## Metrics
- BLEU score is used to evaluate and compare the models
- Results are logged to wandb for visualization