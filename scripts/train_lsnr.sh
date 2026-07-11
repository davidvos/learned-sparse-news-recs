#!/bin/bash
# Train LSNR on MIND or EBNeRD.
#
# Usage:
#   bash scripts/train_lsnr.sh [mind|ebnerd]
#
# All hyperparameters below match the settings reported in the paper.
# Override any value with the corresponding CLI flag (see src/train.py --help).

DATASET=${1:-mind}

uv run python src/train.py \
    --model_type lsnr \
    --dataset "$DATASET" \
    --lambda_n 0.01 \
    --lambda_history 0.001 \
    --lambda_u 0.0001 \
    --topk_sparsity_queries 30 \
    --user_aggregation attention \
    --user_pathway both
