#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4  # matches num_workers in training script
#SBATCH --gres=gpu:1  # requires 1 GPU
#SBATCH --partition=performance
#SBATCH --output=logs/eval_show_tell_%A_%a.out
#SBATCH --error=logs/eval_show_tell_%A_%a.err
#SBATCH --mem=16G


# Run evaluation script
.venv/bin/python eval.py --model show_tell --checkpoint_path $1