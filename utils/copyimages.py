#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser(description="Copy a mask template to a range of frames in two formats.")
    parser.add_argument("--start", type=int, required=True, help="Start frame number (e.g., 70)")
    parser.add_argument("--end", type=int, required=True, help="End frame number (e.g., 80)")
    args = parser.parse_args()

    if args.start > args.end or args.start < 1:
        print("Error: --start must be >= 1 and <= --end")
        sys.exit(1)

    # 🔒 Locked paths
    TEMPLATE_PATH = Path("/home/zheng/mip-splatting-with-mask/data/garden2_mobile/mask/000015.png.png")
    OUTPUT_DIR_1 = Path("/home/zheng/mip-splatting-with-mask/data/garden2_mobile/1")  # .png.png (COLMAP)
    OUTPUT_DIR_2 = Path("/home/zheng/mip-splatting-with-mask/data/garden2_mobile/2")  # .png (standard)

    if not TEMPLATE_PATH.exists():
        print(f"Error: Template file not found: {TEMPLATE_PATH}")
        sys.exit(1)

    OUTPUT_DIR_1.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_2.mkdir(parents=True, exist_ok=True)

    print(f"Copying template to frames {args.start}–{args.end} in two formats...")

    for i in range(args.start, args.end + 1):
        base_name = f"{i:06d}"
        dst1 = OUTPUT_DIR_1 / f"{base_name}.png.png"   # COLMAP double suffix
        dst2 = OUTPUT_DIR_2 / f"{base_name}.png"       # Standard single suffix

        shutil.copy2(TEMPLATE_PATH, dst1)
        shutil.copy2(TEMPLATE_PATH, dst2)

    print("✅ Done!")
    print(f" - COLMAP masks (double .png): {OUTPUT_DIR_1}")
    print(f" - Standard masks (single .png): {OUTPUT_DIR_2}")

if __name__ == "__main__":
    main()