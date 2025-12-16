#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import subprocess
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import torch

"""
Full preprocess pipeline:
1. Convert 360 mp4 → cubemap
2. Crop faces (front, back, left, right)
3. Extract images (fps=1)
4. Generate YOLOv8 segmentation masks (+ optional grow/shrink)
5. Merge to final outputs:
   - images/ : images
   - masks/  : masks with names EXACTLY matching images (e.g., back_back_000001.png)
   - mask/   : COLMAP-style masks with double suffix (e.g., back_back_000001.png.png)
All paths are ABSOLUTE and LOCKED.
"""

# ============================
# ✨ ABSOLUTE PATHS (LOCKED) ✨
# ============================
DATASET_DIR = "/home/zheng/mip-splatting-with-mask/data/garden3/"
INPUT_MP4 = f"{DATASET_DIR}/garden3.mp4"
OUTPUT_IMAGES_DIR = f"{DATASET_DIR}/pre-images"
OUTPUT_MASKS_DIR = f"{DATASET_DIR}/pre-masks"

CUBIC_VIDEO = f"{DATASET_DIR}/garden3_cubic.mp4"

FACE_VIDEOS = {
    "front": f"{DATASET_DIR}/front.mp4",
    "back": f"{DATASET_DIR}/back.mp4",
    "left": f"{DATASET_DIR}/left.mp4",
    "right": f"{DATASET_DIR}/right.mp4",
}

# c3x2 布局的四块裁剪
CROP_FILTERS = {
    "front": "crop=iw/3:ih/2:0:0",
    "back": "crop=iw/3:ih/2:iw/3:0",
    "left": "crop=iw/3:ih/2:iw/3:ih/2",
    "right": "crop=iw/3:ih/2:2*iw/3:ih/2",
}

# ============================
# 可调形态学参数（按需修改）
# ============================
# 模式："dilate"=膨胀(黑色人物区域变大)，"erode"=侵蚀(黑色人物区域变小)
MASK_MORPH_MODE = "dilate"

# 单位像素的半径（核大小约等于 2*radius+1）
MASK_GROW_RADIUS_PX = 16

# 按分辨率比例指定半径（优先生效），例如 0.01 表示 ~1% * min(H,W)
MASK_GROW_RADIUS_REL = 0.0

# 膨胀/侵蚀的“轮数”
MASK_DILATE_ITERATIONS = 5
MASK_ERODE_ITERATIONS = 1

# 可选：填小孔（闭运算：先膨胀后侵蚀）
MASK_CLOSE_HOLES = False
MASK_CLOSE_RADIUS_PX = 3

# 过滤很小的实例（像素阈值）
MIN_INSTANCE_PIXELS = 5000


def run(cmd):
    print("[Running]", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ============================
# Mask helper functions
# ============================
def collect_images(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return [p for p in sorted(root.rglob("*")) if p.suffix.lower() in exts]


def make_mask_path(mask_root: Path, img_root: Path, img_path: Path) -> Path:
    """
    生成与图像路径对应的 mask 路径（注意会附加一个 .png → 形成 .png.png）
    用于 COLMAP 格式（双后缀）。
    """
    rel = img_path.relative_to(img_root)
    return (mask_root / rel).with_suffix(rel.suffix + ".png")  # e.g. xxx.png.png


def resize_bool_mask_to(m: np.ndarray, W: int, H: int) -> np.ndarray:
    n, h0, w0 = m.shape
    if (h0, w0) == (H, W):
        return m
    out = []
    for k in range(n):
        mk = (m[k].astype(np.uint8) * 255)
        mk = cv2.resize(mk, (W, H), interpolation=cv2.INTER_NEAREST)
        out.append(mk > 0)
    return np.stack(out, axis=0)


def make_disk(radius_px: int):
    k = max(1, int(radius_px))
    size = 2 * k + 1  # 奇数核
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def morph_mask(binary_bool: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    根据全局调参对人物二值区域进行膨胀/侵蚀，可选闭运算。
    输入/输出均为 bool(H, W)，True 代表“人物”。
    """
    # 半径
    if MASK_GROW_RADIUS_REL and MASK_GROW_RADIUS_REL > 0:
        r = int(round(min(H, W) * float(MASK_GROW_RADIUS_REL)))
    else:
        r = int(MASK_GROW_RADIUS_PX)

    if r > 0:
        kernel = make_disk(r)
        if MASK_MORPH_MODE.lower() == "erode":
            binary_bool = (
                cv2.erode(binary_bool.astype(np.uint8), kernel, iterations=MASK_ERODE_ITERATIONS) > 0
            )
        else:  # 默认膨胀
            binary_bool = (
                cv2.dilate(binary_bool.astype(np.uint8), kernel, iterations=MASK_DILATE_ITERATIONS) > 0
            )

    if MASK_CLOSE_HOLES:
        kr = max(1, int(MASK_CLOSE_RADIUS_PX))
        k_close = make_disk(kr)
        tmp = cv2.dilate(binary_bool.astype(np.uint8), k_close, iterations=1) > 0
        binary_bool = cv2.erode(tmp.astype(np.uint8), k_close, iterations=1) > 0

    return binary_bool


# ============================
# Main pipeline with conditional steps
# ============================
def main():

    # Prepare directories
    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MASKS_DIR, exist_ok=True)

    for f in FACE_VIDEOS.keys():
        (Path(OUTPUT_IMAGES_DIR) / f).mkdir(parents=True, exist_ok=True)
        (Path(OUTPUT_MASKS_DIR) / f).mkdir(parents=True, exist_ok=True)

    # ----------------------
    # 1) Convert → cubemap  (skip if exists)
    # ----------------------
    if not os.path.exists(CUBIC_VIDEO):
        print("Converting 360 video to cubemap ...")
        run([
            "ffmpeg", "-i", INPUT_MP4,
            "-vf", "v360=e:c3x2,format=yuv420p",
            "-c:v", "hevc_nvenc", "-cq", "25", "-preset", "p7", "-an",
            CUBIC_VIDEO
        ])
    else:
        print(f"[Skip] Cubemap exists: {CUBIC_VIDEO}")

    # ----------------------
    # 2) Crop four faces  (skip each if exists)
    # ----------------------
    for face, crop_filter in CROP_FILTERS.items():
        if not os.path.exists(FACE_VIDEOS[face]):
            print(f"Cropping face: {face}")
            run([
                "ffmpeg", "-i", CUBIC_VIDEO,
                "-vf", crop_filter,
                "-c:v", "hevc_nvenc", "-cq", "25", "-preset", "p7", "-an",
                FACE_VIDEOS[face]
            ])
        else:
            print(f"[Skip] Face video exists: {FACE_VIDEOS[face]}")

    # ----------------------
    # 3) Extract images  (skip if images already present)
    # ----------------------
    for face, video in FACE_VIDEOS.items():
        out_dir = Path(OUTPUT_IMAGES_DIR) / face
        if not any(p.suffix.lower() in {".png", ".jpg", ".jpeg"} for p in out_dir.rglob("*")):
            print(f"Extracting images for {face} ...")
            run([
                "ffmpeg", "-i", video,
                "-vf", "fps=1",
                str(out_dir / f"{face}_%06d.png")
            ])
        else:
            print(f"[Skip] Images exist for {face}: {out_dir}")

    print("Images prepared at:", OUTPUT_IMAGES_DIR)

    # ----------------------
    # 4) YOLO masks (+ morph)  (skip if masks already present)
    # ----------------------
    print("Loading YOLO model…")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8x-seg.pt")

    faces = ["front", "back", "left", "right"]

    for face in faces:
        mask_dir = Path(OUTPUT_MASKS_DIR) / face
        if not any(m.suffix.lower() == ".png" for m in mask_dir.rglob("*")):
            print(f"[Mask] Processing {face}…")
            img_dir = Path(OUTPUT_IMAGES_DIR) / face
            images = collect_images(img_dir)
            if not images:
                print(f"[Warn] No images for {face}, skip masks.")
                continue

            for img_path in images:
                results = model.predict(
                    source=str(img_path),
                    classes=[0],  # person
                    conf=0.25,
                    iou=0.45,
                    device=device,
                    verbose=False
                )
                r = results[0]
                H, W = r.orig_shape
                # 背景=255，人物=0（黑）
                out_mask = np.full((H, W), 255, dtype=np.uint8)

                person_union = None
                if r.masks is not None and len(r.masks) > 0:
                    m = r.masks.data
                    m = (m > 0.5).cpu().numpy()
                    m = resize_bool_mask_to(m, W, H)
                    keep = [mk for mk in m if mk.sum() >= MIN_INSTANCE_PIXELS]
                    if keep:
                        person_union = np.any(np.stack(keep, axis=0), axis=0)

                if person_union is None:
                    person_union = np.zeros((H, W), dtype=bool)

                # 形态学：可调膨胀/侵蚀
                person_union = morph_mask(person_union, H=H, W=W)

                out_mask[person_union] = 0

                out_path_colmap = make_mask_path(mask_dir, img_dir, img_path)  # 将保存为 *.png.png
                out_path_colmap.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path_colmap), out_mask)
        else:
            print(f"[Skip] Pre-masks exist for {face}: {mask_dir}")

    print("Pre-masks saved at:", OUTPUT_MASKS_DIR)

    # ----------------------
    # 5) Merge to final outputs
    #    - images/ : f"{face}_{img.name}"
    #    - masks/  : SAME NAME as images (single .png)
    #    - mask/   : COLMAP-style, double suffix (add an extra .png)
    # ----------------------
    FINAL_IMG = f"{DATASET_DIR}/images"
    FINAL_MASK_STD = f"{DATASET_DIR}/masks"  # 与 images 名字一致（单 .png）
    FINAL_MASK_COLMAP = f"{DATASET_DIR}/mask"  # COLMAP 格式（双 .png）

    Path(FINAL_IMG).mkdir(exist_ok=True)
    Path(FINAL_MASK_STD).mkdir(exist_ok=True)
    Path(FINAL_MASK_COLMAP).mkdir(exist_ok=True)

    faces_list = ["front", "back", "left", "right"]

    # ---- merge images
    for face in faces_list:
        src_imgs = Path(OUTPUT_IMAGES_DIR) / face
        for img in src_imgs.rglob("*.png"):
            dst = Path(FINAL_IMG) / f"{face}_{img.name}"  # e.g., back_back_000001.png
            if not dst.exists():
                dst.write_bytes(img.read_bytes())

    # ---- merge masks (两套命名)
    for face in faces_list:
        img_dir = Path(OUTPUT_IMAGES_DIR) / face
        mask_dir = Path(OUTPUT_MASKS_DIR) / face

        # 逐张图像驱动：保证两种命名都能准确对应
        for img in img_dir.rglob("*.png"):
            # 源 mask（pre-masks 的 COLMAP 双后缀）
            src_mask_colmap = make_mask_path(mask_dir, img_dir, img)  # .../pre-masks/face/face_xxx.png.png
            if not src_mask_colmap.exists():
                # 如果不存在，跳过（或打印警告）
                print(f"[Warn] Missing pre-mask for {img}")
                continue

            base_img_name = f"{face}_{img.name}"           # back_back_000001.png
            base_img_name_colmap = base_img_name + ".png"  # back_back_000001.png.png

            # 1) 写入 masks/（与 images 同名：单 .png）
            dst_std = Path(FINAL_MASK_STD) / base_img_name
            if not dst_std.exists():
                dst_std.write_bytes(src_mask_colmap.read_bytes())

            # 2) 写入 mask/（COLMAP 双后缀）
            dst_colmap = Path(FINAL_MASK_COLMAP) / base_img_name_colmap
            if not dst_colmap.exists():
                dst_colmap.write_bytes(src_mask_colmap.read_bytes())

    print("Final folders:")
    print(" images :", FINAL_IMG)
    print(" masks  :", FINAL_MASK_STD, "   (names match images, e.g., back_back_000001.png)")
    print(" mask   :", FINAL_MASK_COLMAP, " (COLMAP style, e.g., back_back_000001.png.png)")


if __name__ == "__main__":
    main()
