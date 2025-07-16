#!/bin/bash

# 设置只使用前4个GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 使用4个GPU进行训练
accelerate launch --num_processes=4 train.py \
  --report-to="tensorboard" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --model="SiT-XL/2" \
  --enc-type="DCT" \
  --proj-coeff=0.5 \
  --encoder-depth=8 \
  --output-dir="exps" \
  --exp-name="dct-film-full-70k-4gpu" \
  --data-dir="full_dataset" \
  --batch-size=256 \
  --epochs=100 \
  --max-train-steps=400000 \
  --checkpointing-steps=10000 \
  --num-workers=8 