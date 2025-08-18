#!/usr/bin/env python3
"""
ImageNet 256x256 数据集预处理脚本

这个脚本将完成以下任务：
1. 创建正确的目录结构 (images/ 和 vae-sd/)
2. 复制和整理图像文件到flat结构
3. 生成数据集JSON标签文件
4. 执行VAE编码

使用方法:
python prepare_imagenet_dataset.py --source /path/to/imagenet256 --dest /path/to/processed_data
"""

import argparse
import json
import os
import shutil
import glob
from pathlib import Path
from tqdm import tqdm
import logging

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('dataset_preparation.log')
        ]
    )

def create_directory_structure(dest_dir):
    """创建目标目录结构"""
    dest_path = Path(dest_dir)
    images_dir = dest_path / "images"
    vae_dir = dest_path / "vae-sd"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    vae_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Created directories: {images_dir}, {vae_dir}")
    return images_dir, vae_dir

def load_label_mapping(source_dir):
    """加载标签映射"""
    label_file = Path(source_dir) / "label_mapping.json"
    if label_file.exists():
        with open(label_file, 'r') as f:
            label_mapping = json.load(f)
        # 转换为class_name -> label_id的映射
        name_to_id = {}
        for label_id, class_name in label_mapping.items():
            # 取第一个逗号前的部分作为目录名的前缀匹配
            primary_name = class_name.split(',')[0].strip()
            name_to_id[primary_name.lower().replace(' ', '_')] = int(label_id)
        return name_to_id
    else:
        logging.warning("No label_mapping.json found, will use folder-based indexing")
        return None

def copy_and_flatten_images(source_dir, dest_images_dir, label_mapping=None):
    """
    复制图像文件到flat结构并生成标签
    
    Args:
        source_dir: ImageNet源目录 (包含train/子目录)
        dest_images_dir: 目标图像目录
        label_mapping: 类名到标签ID的映射
    
    Returns:
        labels: 包含[filename, label_id]的列表
    """
    source_path = Path(source_dir)
    train_dir = source_path / "train"
    
    if not train_dir.exists():
        raise ValueError(f"Train directory not found: {train_dir}")
    
    # 获取所有类别目录
    class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
    class_dirs.sort()
    
    labels = []
    copied_count = 0
    
    logging.info(f"Found {len(class_dirs)} class directories")
    
    for class_idx, class_dir in enumerate(tqdm(class_dirs, desc="Processing classes")):
        class_name = class_dir.name
        
        # 确定标签ID
        if label_mapping:
            # 尝试多种匹配方式
            label_id = None
            # 1. 直接匹配
            if class_name.lower() in label_mapping:
                label_id = label_mapping[class_name.lower()]
            else:
                # 2. 尝试匹配类名的各个部分
                for key in label_mapping.keys():
                    if key in class_name.lower() or class_name.lower() in key:
                        label_id = label_mapping[key]
                        break
            
            if label_id is None:
                logging.warning(f"No label found for class: {class_name}, using index {class_idx}")
                label_id = class_idx
        else:
            label_id = class_idx
        
        # 复制类别下的所有图像
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPEG"))
        
        for img_file in tqdm(image_files, desc=f"Class {class_name}", leave=False):
            # 生成新的文件名：原文件名 (保持唯一性)
            new_filename = f"{class_name}_{img_file.name}"
            dest_path = dest_images_dir / new_filename
            
            # 避免重复复制
            if not dest_path.exists():
                shutil.copy2(img_file, dest_path)
                copied_count += 1
            
            labels.append([new_filename, label_id])
    
    logging.info(f"Copied {copied_count} images to {dest_images_dir}")
    return labels

def generate_dataset_json(labels, dest_images_dir):
    """生成dataset.json文件"""
    dataset_dict = {"labels": labels}
    json_path = dest_images_dir / "dataset.json"
    
    with open(json_path, 'w') as f:
        json.dump(dataset_dict, f, indent=2)
    
    logging.info(f"Generated dataset.json with {len(labels)} entries at {json_path}")

def main():
    parser = argparse.ArgumentParser(description="Prepare ImageNet dataset for REPA training")
    parser.add_argument("--source", type=str, required=True, 
                        help="Source ImageNet directory (containing train/ subdirectory)")
    parser.add_argument("--dest", type=str, required=True,
                        help="Destination directory for processed dataset")
    parser.add_argument("--copy-images", action="store_true", default=True,
                        help="Copy and flatten images (default: True)")
    parser.add_argument("--generate-json", action="store_true", default=True,
                        help="Generate dataset.json (default: True)")
    
    args = parser.parse_args()
    
    setup_logging()
    
    logging.info("Starting ImageNet dataset preparation...")
    logging.info(f"Source: {args.source}")
    logging.info(f"Destination: {args.dest}")
    
    # 创建目录结构
    images_dir, vae_dir = create_directory_structure(args.dest)
    
    if args.copy_images:
        # 加载标签映射
        label_mapping = load_label_mapping(args.source)
        
        # 复制并flat化图像
        labels = copy_and_flatten_images(args.source, images_dir, label_mapping)
        
        if args.generate_json:
            # 生成dataset.json
            generate_dataset_json(labels, images_dir)
    
    logging.info("Dataset preparation completed!")
    logging.info(f"Next steps:")
    logging.info(f"1. Run VAE encoding: cd preprocessing && python dataset_tools.py encode --source={images_dir} --dest={vae_dir}")
    logging.info(f"2. Generate FID stats: python -m tools.generate_fid_stats {images_dir} {args.dest}/fid_stats.npz")

if __name__ == "__main__":
    main()
