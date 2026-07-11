#!/bin/bash

uv run python src/train.py \
    --model_type lsnr \
    --dataset mind \
    --experiment_name lsnr_dense_only_mind \
    --lambda_n 0.01 \
    --lambda_history 0.001 \
    --lambda_u 0.0001 \
    --user_aggregation attention \
    --user_pathway dense
