#!/bin/bash

# 使用8个GPU进行训练（如果可用）
# 如果只有单GPU，将num_processes改为1

accelerate launch --num_processes=8 train.py \
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
  --exp-name="dct-film-full-70k" \
  --data-dir="full_dataset" \
  --batch-size=256 \
  --epochs=100 \
  --max-train-steps=400000 \
  --checkpointing-steps=10000 \
  --num-workers=8 