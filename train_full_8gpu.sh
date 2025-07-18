#!/bin/bash

# 8GPU训练脚本 - 使用nohup防止意外中断
export CUDA_VISIBLE_DEVICES=4,5,6,7

# 创建日志目录
mkdir -p logs

# 启动训练，输出重定向到日志文件
nohup accelerate launch \
  --num_processes=4 \
  --main_process_port=29500 \
  train.py \
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
  --exp-name="dct-film-full-70k-8gpu" \
  --data-dir="full_dataset" \
  --batch-size=128 \
  --epochs=100 \
  --max-train-steps=50000 \
  --checkpointing-steps=5000 \
  --num-workers=16 \
  --proj-type="cosine" \
  --gradient-accumulation-steps=1 \
  > logs/train_4gpu_l2_multi.log 2>&1 &

echo "训练已在后台启动！"
echo "查看训练日志: tail -f logs/train_8gpu.log"
echo "查看GPU使用: watch -n 1 nvidia-smi"
echo "查看tensorboard: tensorboard --logdir=exps/dct-film-full-70k-8gpu" 