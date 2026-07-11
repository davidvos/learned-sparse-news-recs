#!/bin/bash
# Train STAR baseline on MIND or EBNeRD.
#
# Usage:
#   bash scripts/train_star.sh [mind|ebnerd]

DATASET=${1:-mind}

uv run python src/train.py \
    --model_type star \
    --dataset "$DATASET" \
    --lambda_n 0.01
