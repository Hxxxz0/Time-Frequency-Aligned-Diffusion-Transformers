#!/usr/bin/env python3
"""
CIFAR-10数据集预处理脚本
将32x32的CIFAR-10图像插值到256x256并转换为训练格式
"""

import os
import json
import numpy as np
import PIL.Image
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
import shutil

# CIFAR-10类别名称（中文）
CIFAR10_CLASSES = {
    0: '飞机',    # airplane
    1: '汽车',    # automobile  
    2: '鸟',      # bird
    3: '猫',      # cat
    4: '鹿',      # deer
    5: '狗',      # dog
    6: '青蛙',    # frog
    7: '马',      # horse
    8: '船',      # ship
    9: '卡车'     # truck
}

def download_cifar10(data_dir='./cifar10_raw'):
    """下载CIFAR-10数据集"""
    print("正在下载CIFAR-10数据集...")
    
    # 下载训练和测试集
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=None
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=None
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    return train_dataset, test_dataset

def resize_and_save_images(dataset, output_dir, target_size=(256, 256), split_name="train"):
    """将CIFAR-10图像插值到目标尺寸并保存"""
    
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # 用于高质量插值的transform
    resize_transform = transforms.Compose([
        transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.ToPILImage()
    ])
    
    labels_data = []
    
    print(f"正在处理{split_name}集，插值到{target_size[0]}x{target_size[1]}...")
    
    for idx, (image, label) in tqdm(enumerate(dataset), total=len(dataset)):
        # 确保image是PIL Image格式
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)
        
        # 插值到目标尺寸
        resized_image = resize_transform(image)
        
        # 保存图像
        idx_str = f'{idx:08d}'
        img_filename = f'{idx_str[:5]}/img{idx_str}.png'
        img_path = os.path.join(images_dir, img_filename)
        
        # 创建子目录
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        
        # 保存为PNG格式
        resized_image.save(img_path, format='PNG', compress_level=0, optimize=False)
        
        # 记录标签信息
        labels_data.append([img_filename, int(label)])
    
    # 保存标签文件
    dataset_metadata = {
        'labels': labels_data,
        'classes': CIFAR10_CLASSES,
        'resolution': target_size,
        'split': split_name,
        'total_images': len(labels_data)
    }
    
    with open(os.path.join(images_dir, 'dataset.json'), 'w', encoding='utf-8') as f:
        json.dump(dataset_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"已保存{len(labels_data)}张{target_size[0]}x{target_size[1]}图像到 {images_dir}")
    return images_dir

def run_vae_encoding(images_dir, output_dir):
    """使用现有的dataset_tools.py进行VAE编码"""
    vae_dir = os.path.join(output_dir, 'vae-sd')
    
    print("正在进行VAE编码...")
    print(f"输入目录: {images_dir}")
    print(f"输出目录: {vae_dir}")
    
    # 调用dataset_tools.py的encode命令
    import subprocess
    
    # 检查当前工作目录，决定正确的工作目录
    current_dir = os.getcwd()
    if current_dir.endswith('preprocessing'):
        work_dir = '.'
    else:
        work_dir = 'preprocessing'
    
    cmd = [
        'python', 'dataset_tools.py', 'encode',
        '--source', os.path.abspath(images_dir),
        '--dest', os.path.abspath(vae_dir)
    ]
    
    try:
        result = subprocess.run(cmd, cwd=work_dir, check=True, capture_output=True, text=True)
        print("VAE编码完成！")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"VAE编码失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False
    
    return True

def create_cifar10_dataset(target_size=(256, 256), use_both_splits=True, output_dir='cifar10_processed'):
    """完整的CIFAR-10数据集创建流程"""
    
    print("=" * 60)
    print("CIFAR-10数据集预处理开始")
    print("=" * 60)
    
    # 步骤1: 下载数据
    train_dataset, test_dataset = download_cifar10()
    
    # 步骤2: 选择使用的数据
    if use_both_splits:
        # 合并训练集和测试集
        print("合并训练集和测试集...")
        all_images = []
        all_labels = []
        
        for img, label in train_dataset:
            all_images.append(img)
            all_labels.append(label)
            
        for img, label in test_dataset:
            all_images.append(img)
            all_labels.append(label)
        
        # 创建合并数据集
        class CombinedDataset:
            def __init__(self, images, labels):
                self.images = images
                self.labels = labels
            
            def __len__(self):
                return len(self.images)
            
            def __getitem__(self, idx):
                return self.images[idx], self.labels[idx]
        
        combined_dataset = CombinedDataset(all_images, all_labels)
        print(f"合并后数据集大小: {len(combined_dataset)}")
        
        # 步骤3: 插值并保存图像
        images_dir = resize_and_save_images(
            combined_dataset, output_dir, target_size, "combined"
        )
    else:
        # 只使用训练集
        images_dir = resize_and_save_images(
            train_dataset, output_dir, target_size, "train"
        )
    
    # 步骤4: VAE编码
    success = run_vae_encoding(images_dir, output_dir)
    
    if success:
        print("=" * 60)
        print("CIFAR-10数据集预处理完成！")
        print("=" * 60)
        print(f"处理后的数据集位置: {output_dir}")
        print(f"图像目录: {images_dir}")
        print(f"VAE编码目录: {os.path.join(output_dir, 'vae-sd')}")
        print("\n可以使用以下命令开始训练:")
        print(f"--data-dir={output_dir}")
    else:
        print("VAE编码失败，请检查错误信息")

def main():
    parser = argparse.ArgumentParser(description='CIFAR-10数据集预处理')
    parser.add_argument('--output-dir', default='cifar10_processed', 
                        help='输出目录 (默认: cifar10_processed)')
    parser.add_argument('--resolution', default='256x256', 
                        help='目标分辨率 (默认: 256x256)')
    parser.add_argument('--train-only', action='store_true',
                        help='只使用训练集（默认使用训练集+测试集）')
    
    args = parser.parse_args()
    
    # 解析分辨率
    width, height = map(int, args.resolution.split('x'))
    target_size = (width, height)
    
    print(f"目标分辨率: {width}x{height}")
    print(f"输出目录: {args.output_dir}")
    print(f"使用数据: {'仅训练集' if args.train_only else '训练集+测试集'}")
    
    create_cifar10_dataset(
        target_size=target_size,
        use_both_splits=not args.train_only,
        output_dir=args.output_dir
    )

if __name__ == '__main__':
    main() 