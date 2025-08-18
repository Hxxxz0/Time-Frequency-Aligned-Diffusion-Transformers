#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download & export benjamin-paine/imagenet-1k-256x256 to ImageFolder layout.

Output layout:
  <out_dir>/
    train/<class_slug>/*.jpg
    validation/<class_slug>/*.jpg
    test/unknown/*.jpg           # test split 无标签，统一放到 unknown 目录
Also writes: <out_dir>/label_mapping.json  (index -> human-readable class name)

Usage:
  python save_imagenet256.py --out /path/to/imagenet256 --splits train val test --num-workers 8
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import List

from datasets import load_dataset
from tqdm import tqdm

def slugify(name: str) -> str:
    # 将类名转换为安全的目录名（去掉逗号/括号/多空格等）
    name = name.lower()
    name = name.replace("&", "and")
    name = re.sub(r"[(),]", "", name)
    name = re.sub(r"[^a-z0-9\-\s_./]+", "", name)
    name = re.sub(r"\s+", "_", name).strip("_")
    return name[:80]  # 限长，避免奇怪文件系统限制

def save_split(ds, out_dir: Path, split: str, class_names: List[str]):
    out_dir = out_dir / split
    unknown_dir = out_dir / "unknown"
    out_dir.mkdir(parents=True, exist_ok=True)
    if split == "test":
        unknown_dir.mkdir(parents=True, exist_ok=True)

    # 预先创建所有类目录（train / validation）
    if split != "test":
        for idx, cname in enumerate(class_names):
            (out_dir / slugify(cname)).mkdir(parents=True, exist_ok=True)

    # 逐样本保存
    # 注意：dataset[i]["image"] 返回的是解码后的 PIL.Image
    for i, ex in enumerate(tqdm(ds, desc=f"Saving {split}", unit="img")):
        img = ex["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        label = ex["label"]

        if split == "test" or label == -1:
            save_dir = unknown_dir
        else:
            class_name = class_names[label]
            save_dir = out_dir / slugify(class_name)

        # 有些系统目录中文件太多会卡，这里按每类平铺，文件名用自增编号
        fname = f"{i:09d}.jpg"
        fpath = save_dir / fname
        # 断点续跑：已存在就跳过
        if fpath.exists():
            continue
        try:
            img.save(fpath, format="JPEG", quality=95, optimize=True)
        except Exception as e:
            # 个别损坏样本直接跳过
            print(f"[WARN] failed saving {split} idx={i}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=str, help="Output root directory")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"],
                        choices=["train", "validation", "test", "val", "all"],
                        help="Which splits to export")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="datasets 解码/后处理进程数（I/O 为主，1-8 够用）")
    parser.add_argument("--hf-cache", type=str, default=None,
                        help="Optional: set HF_DATASETS_CACHE dir")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 规范化 split 名
    splits = args.splits
    if "all" in splits:
        splits = ["train", "validation", "test"]
    splits = ["validation" if s == "val" else s for s in splits]

    if args.hf_cache:
        os.environ["HF_DATASETS_CACHE"] = args.hf_cache  # 可选：指定缓存

    # 加载一个 split 拿到标签名（训练/验证都有 label 名字）
    ds_info = load_dataset("benjamin-paine/imagenet-1k-256x256", split="train", streaming=False)
    # label -> 人类可读类名（见官方卡片：labels 按 synset 排序映射）：
    # https://huggingface.co/datasets/benjamin-paine/imagenet-1k-256x256
    features = ds_info.features
    if "label" not in features or features["label"].names is None:
        # 兜底：用 int2str
        int2str = ds_info.features["label"].int2str
        class_names = [int2str(i) for i in range(1000)]
    else:
        class_names = list(features["label"].names)

    # 保存一份索引->类名映射
    with open(out_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump({i: name for i, name in enumerate(class_names)}, f, ensure_ascii=False, indent=2)

    for split in splits:
        ds = load_dataset("benjamin-paine/imagenet-1k-256x256", split=split, streaming=False)
        # datasets 已经把 image 列解码为 PIL.Image；无需再 resize（本身即 256x256）
        save_split(ds, out_dir, split, class_names)

    print(f"✅ Done. Exported to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
