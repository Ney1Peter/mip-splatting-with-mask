import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ================== 配置路径 ==================
IMAGE_DIR = "/home/zheng/mip-splatting-with-mask/data/garden3/images"
MASK_OUTPUT_DIR = "/home/zheng/mip-splatting-with-mask/data/garden3/mask_test"

# ================== 参数设置 ==================
MIN_VALID_KEYPOINTS = 10
KEYPOINT_CONF_THRESHOLD = 0.5
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

# ================== 初始化 ==================
os.makedirs(MASK_OUTPUT_DIR, exist_ok=True)
print("Loading YOLOv8-pose model...")
model = YOLO("yolov8n-pose.pt")

image_paths = [
    p for p in Path(IMAGE_DIR).iterdir()
    if p.suffix.lower() in IMAGE_EXTENSIONS
]
print(f"Found {len(image_paths)} image(s) in {IMAGE_DIR}")

# ================== 处理每张图像 ==================
for img_path in sorted(image_paths):
    print(f"Processing: {img_path.name}")

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  Failed to read image: {img_path}")
        continue

    h, w = img.shape[:2]
    # 初始化为全白（背景保留）
    mask = np.full((h, w), 255, dtype=np.uint8)

    results = model(img, verbose=False)
    has_real_person = False

    if results and len(results[0].boxes) > 0:
        for i in range(len(results[0].boxes)):
            kpts = results[0].keypoints[i].data.cpu().numpy()
            if kpts.ndim == 3:
                kpts = kpts.squeeze(0)
            elif kpts.ndim != 2:
                continue

            valid_count = np.sum(kpts[:, 2] > KEYPOINT_CONF_THRESHOLD)
            if valid_count >= MIN_VALID_KEYPOINTS:
                has_real_person = True
                box = results[0].boxes[i].xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = np.clip(box, 0, [w, h, w, h])
                mask[y1:y2, x1:x2] = 0  # 人物区域设为黑色（表示要移除）

    mask_filename = img_path.with_suffix('.png').name
    mask_path = os.path.join(MASK_OUTPUT_DIR, mask_filename)
    cv2.imwrite(mask_path, mask)

    status = "检测到真人 → 白色区域" if has_real_person else "无真人（可能为雕像）→ 全黑"
    print(f"  → {status}, saved to {mask_path}")

print("All images processed.")