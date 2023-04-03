#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python generate.py \
    --exp_name CAIN_fin \
    --dataset custom \
    --data_root example \
    --img_fmt png \
    --batch_size 32 \
    --test_batch_size 1 \
    --model cain \
    --depth 3 \
    --loss 1*L1 \
    --resume \
    --mode test