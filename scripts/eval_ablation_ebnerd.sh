#!/bin/bash

echo "=== EBNeRD sparse_only (u_lex) | cold0 ==="
uv run python src/evaluate.py \
    --model_type lsnr --dataset ebnerd --split val \
    --experiment_name lsnr_sparse_only_ebnerd \
    --user_pathway sparse --cold_threshold 0

echo "=== EBNeRD sparse_only (u_lex) | cold1 ==="
uv run python src/evaluate.py \
    --model_type lsnr --dataset ebnerd --split val \
    --experiment_name lsnr_sparse_only_ebnerd \
    --user_pathway sparse --cold_threshold 1

echo "=== EBNeRD dense_only (u_sem) | cold0 ==="
uv run python src/evaluate.py \
    --model_type lsnr --dataset ebnerd --split val \
    --experiment_name lsnr_dense_only_ebnerd \
    --user_pathway dense --cold_threshold 0

echo "=== EBNeRD dense_only (u_sem) | cold1 ==="
uv run python src/evaluate.py \
    --model_type lsnr --dataset ebnerd --split val \
    --experiment_name lsnr_dense_only_ebnerd \
    --user_pathway dense --cold_threshold 1

echo "=== EBNeRD combined (both) | cold0 ==="
uv run python src/evaluate.py \
    --model_type lsnr --dataset ebnerd --split val \
    --experiment_name lsnr_combined_ebnerd \
    --user_pathway both --cold_threshold 0

echo "=== EBNeRD combined (both) | cold1 ==="
uv run python src/evaluate.py \
    --model_type lsnr --dataset ebnerd --split val \
    --experiment_name lsnr_combined_ebnerd \
    --user_pathway both --cold_threshold 1
