#!/bin/bash

# 4GPU训练脚本 - 按照README标准配置
export CUDA_VISIBLE_DEVICES=3,4,5,6

# 创建日志目录
mkdir -p logs

# 启动训练，使用torchrun多GPU启动
nohup /data/wenshuo/miniconda3/envs/repa/bin/torchrun --nproc_per_node=4 train.py \
  --report-to="tensorboard" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --model="SiT-XL/2" \
  --enc-type="DCT" \
  --proj-coeff=0.0 \
  --encoder-depth=8 \
  --output-dir="exps" \
  --exp-name="dct-film-full-70k-4gpu_new" \
  --data-dir="REPA/full_dataset" \
  --resume-from="REPA/exps/dct-film-full-70k-4gpu_D_no_film_abl/checkpoints/0010000.pt" \
  --batch-size=128 \
  --epochs=100 \
  --max-train-steps=400000 \
  --checkpointing-steps=5000 \
  --num-workers=16 \
  --proj-type="l2" \
  --gradient-accumulation-steps=1 \
  --enable-fid-eval \
  --fid-samples=5000 \
  > REPA/logs/train_4gpu_l2_multi_new.log 2>&1 &

echo "训练已在后台启动！"
echo "查看训练日志: tail -f REPA/logs/train_4gpu_l2_multi_new.log"
echo "查看GPU使用: watch -n 1 nvidia-smi"
echo "查看tensorboard: tensorboard --logdir=exps/dct-film-full-70k-4gpu"
echo "提示：如果遇到 'no supported trackers are currently installed' 警告，请运行 'pip install tensorboard'。"
echo ""
echo "FID评估相关提示："
echo "1. 训练将在每个checkpoint时自动计算FID分数"
echo "2. 如果提示缺少FID统计文件，请先运行以下命令："
echo "   python -m tools.generate_fid_stats REPA/full_dataset/images REPA/full_dataset/fid_stats.npz"
echo "3. FID日志将保存在实验目录下的 fid_log.txt 文件中" 