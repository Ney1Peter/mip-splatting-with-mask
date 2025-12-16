#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import re
from pathlib import Path
import argparse

def extract_number(filename: str) -> int:
    """从文件名如 '000001.png' 或 '000001.png.png' 中提取数字部分"""
    # 匹配开头的数字（至少1位）
    match = re.match(r'^(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Cannot extract number from filename: {filename}")

def rename_in_directory(dir_path: Path, start_index: int, is_double_suffix: bool = False):
    """重命名目录中的图片文件"""
    if not dir_path.exists():
        print(f"Directory not found, skipped: {dir_path}")
        return

    # 收集所有图片文件
    files = []
    if is_double_suffix:
        # mask/ 目录：匹配 *.png.png
        for f in dir_path.iterdir():
            if f.is_file() and f.suffix == '.png' and f.name.count('.png') >= 2:
                files.append(f)
    else:
        # images/ 和 masks/：匹配 *.png（但不是 *.png.png）
        for f in dir_path.iterdir():
            if f.is_file() and f.suffix == '.png' and not f.name.endswith('.png.png'):
                files.append(f)

    if not files:
        print(f"No valid image files in {dir_path}, skipped.")
        return

    # 按数字排序
    try:
        files.sort(key=lambda x: extract_number(x.name))
    except ValueError as e:
        print(f"Error sorting files in {dir_path}: {e}")
        return

    print(f"Renaming {len(files)} files in {dir_path} starting from {start_index:06d}...")

    # 临时重命名（避免覆盖）
    temp_files = []
    for i, f in enumerate(files):
        new_num = start_index + i
        if is_double_suffix:
            new_name = f"{new_num:06d}.png.png"
        else:
            new_name = f"{new_num:06d}.png"
        temp_name = f"__temp_{new_name}"
        temp_path = dir_path / temp_name
        f.rename(temp_path)
        temp_files.append((temp_path, dir_path / new_name))

    # 从临时名改为最终名
    for temp_path, final_path in temp_files:
        temp_path.rename(final_path)

    print(f"Done: {dir_path}")

def main():
    parser = argparse.ArgumentParser(description="Rename image sequences in images/, masks/, mask/ directories.")
    parser.add_argument("dataset_dir", type=str, help="Path to dataset directory (e.g., /home/.../garden2_mobile)")
    parser.add_argument("--start", type=int, required=True, help="Starting frame number (e.g., 170)")
    args = parser.parse_args()

    dataset = Path(args.dataset_dir)
    start = args.start

    if not dataset.is_dir():
        print(f"Dataset directory not found: {dataset}")
        sys.exit(1)

    # 定义三个目录
    images_dir = dataset / "images"
    masks_std_dir = dataset / "masks"      # single .png
    masks_colmap_dir = dataset / "mask"    # double .png.png

    print(f"Renaming sequences starting from {start:06d} in:\n  {dataset}\n")

    # 重命名三个目录
    rename_in_directory(images_dir, start, is_double_suffix=False)
    rename_in_directory(masks_std_dir, start, is_double_suffix=False)
    rename_in_directory(masks_colmap_dir, start, is_double_suffix=True)

    print("\n All renaming completed!")

if __name__ == "__main__":
    main()