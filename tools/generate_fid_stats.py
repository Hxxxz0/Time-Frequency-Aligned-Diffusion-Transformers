#!/usr/bin/env python3
"""
生成FID计算所需的统计文件

用法:
python tools/generate_fid_stats.py /path/to/real/images /path/to/output/fid_stats.npz
"""

import argparse
import os
from tools.fid_score import save_statistics_of_path


def main():
    parser = argparse.ArgumentParser(description="Generate FID statistics file")
    parser.add_argument("images_path", type=str, help="Path to real images directory")
    parser.add_argument("output_path", type=str, help="Output path for FID statistics (.npz file)")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for processing")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of workers for data loading")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.images_path):
        print(f"错误: 图像目录不存在: {args.images_path}")
        return
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output_path)
    if output_dir:  # 只有当目录不为空时才创建
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"开始为 {args.images_path} 生成FID统计文件...")
    print(f"输出文件: {args.output_path}")
    
    save_statistics_of_path(
        args.images_path,
        args.output_path,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"FID统计文件已保存到: {args.output_path}")


if __name__ == "__main__":
    main()