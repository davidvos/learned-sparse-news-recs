#!/bin/bash

echo "=== MIND sparse_only (u_lex) | cold0 ==="
uv run python src/evaluate.py \
    --model_type lsnr --dataset mind --split val \
    --experiment_name lsnr_sparse_only_mind \
    --user_pathway sparse --cold_threshold 0

echo "=== MIND sparse_only (u_lex) | cold1 ==="
uv run python src/evaluate.py \
    --model_type lsnr --dataset mind --split val \
    --experiment_name lsnr_sparse_only_mind \
    --user_pathway sparse --cold_threshold 1

echo "=== MIND dense_only (u_sem) | cold0 ==="
uv run python src/evaluate.py \
    --model_type lsnr --dataset mind --split val \
    --experiment_name lsnr_dense_only_mind \
    --user_pathway dense --cold_threshold 0

echo "=== MIND dense_only (u_sem) | cold1 ==="
uv run python src/evaluate.py \
    --model_type lsnr --dataset mind --split val \
    --experiment_name lsnr_dense_only_mind \
    --user_pathway dense --cold_threshold 1

echo "=== MIND combined (both) | cold0 ==="
uv run python src/evaluate.py \
    --model_type lsnr --dataset mind --split val \
    --experiment_name lsnr_combined_mind \
    --user_pathway both --cold_threshold 0

echo "=== MIND combined (both) | cold1 ==="
uv run python src/evaluate.py \
    --model_type lsnr --dataset mind --split val \
    --experiment_name lsnr_combined_mind \
    --user_pathway both --cold_threshold 1
