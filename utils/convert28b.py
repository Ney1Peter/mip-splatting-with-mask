#!/usr/bin/env python3
import cv2
from pathlib import Path

# 配置
images_dir = Path("/home/zheng/mip-splatting-with-mask/data/tombstone_mobile/images")

# 遍历所有 PNG
for img_path in sorted(images_dir.glob("*.png")):
    print(f"Processing: {img_path.name}")
    
    # 读取（OpenCV 自动处理 16-bit）
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"  Failed to read {img_path}")
        continue

    # 如果是 16-bit，转换为 8-bit
    if img.dtype == 'uint16':
        print(f"  Converting from 16-bit to 8-bit")
        img = (img.astype('float32') / 65535.0 * 255.0).astype('uint8')
    elif img.dtype != 'uint8':
        print(f"  Unexpected dtype: {img.dtype}, converting to uint8")
        img = img.astype('uint8')

    # 确保是 3 通道（防止灰度）
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        # RGBA → RGB（白底）
        alpha = img[:, :, 3] / 255.0
        img_rgb = img[:, :, :3]
        white_bg = 255 * (1 - alpha[..., None])
        img = (img_rgb * alpha[..., None] + white_bg).astype('uint8')

    # 保存（强制 8-bit，non-interlaced）
    cv2.imwrite(str(img_path), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])

print("All images converted to 8-bit RGB PNG.")