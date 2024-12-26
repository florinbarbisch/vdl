#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4  # matches num_workers in training script
#SBATCH --gres=gpu:1  # requires 1 GPU
#SBATCH --partition=performance
#SBATCH --output=logs/show_tell_%A_%a.out
#SBATCH --error=logs/show_tell_%A_%a.err
#SBATCH --mem=16G

# Create necessary directories
mkdir -p logs
mkdir -p checkpoints

# Run training script
.venv/bin/python src/train.py --model show_tell