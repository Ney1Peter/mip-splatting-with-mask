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
2. Crop 6 faces
3. Extract images (fps=1)
4. Generate YOLOv8 segmentation masks
All paths are ABSOLUTE and LOCKED.
"""

# ============================
# ✨ ABSOLUTE PATHS (LOCKED) ✨
# ============================
DATASET_DIR = "/home/zheng/mip-splatting-with-mask/test"
INPUT_MP4 = f"{DATASET_DIR}/lake.mp4"
OUTPUT_IMAGES_DIR = f"{DATASET_DIR}/pre-images"
OUTPUT_MASKS_DIR = f"{DATASET_DIR}/pre-masks"

CUBIC_VIDEO = f"{DATASET_DIR}/lake_cubic.mp4"

FACE_VIDEOS = {
    "front": f"{DATASET_DIR}/front.mp4",
    "back": f"{DATASET_DIR}/back.mp4",
    "up": f"{DATASET_DIR}/up.mp4",
    "down": f"{DATASET_DIR}/down.mp4",
    "left": f"{DATASET_DIR}/left.mp4",
    "right": f"{DATASET_DIR}/right.mp4",
}

CROP_FILTERS = {
    "front": "crop=iw/3:ih/2:0:0",
    "back": "crop=iw/3:ih/2:iw/3:0",
    "up": "crop=iw/3:ih/2:2*iw/3:0",
    "down": "crop=iw/3:ih/2:0:ih/2",
    "left": "crop=iw/3:ih/2:iw/3:ih/2",
    "right": "crop=iw/3:ih/2:2*iw/3:ih/2",
}


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
    rel = img_path.relative_to(img_root)
    return (mask_root / rel).with_suffix(rel.suffix + ".png")


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


# ============================
# Main pipeline
# ============================
def main():

    # Prepare directories
    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MASKS_DIR, exist_ok=True)

    for f in FACE_VIDEOS.keys():
        os.makedirs(f"{OUTPUT_IMAGES_DIR}/{f}", exist_ok=True)
        os.makedirs(f"{OUTPUT_MASKS_DIR}/{f}", exist_ok=True)

    # ----------------------
    # 1. Convert → cubemap
    # ----------------------
    run([
        "ffmpeg", "-i", INPUT_MP4,
        "-vf", "v360=e:c3x2,format=yuv420p",
        "-c:v", "hevc_nvenc", "-cq", "25", "-preset", "p7", "-an",
        CUBIC_VIDEO
    ])

    # ----------------------
    # 2. Crop 6 faces
    # ----------------------
    for face, crop_filter in CROP_FILTERS.items():
        run([
            "ffmpeg", "-i", CUBIC_VIDEO,
            "-vf", crop_filter,
            "-c:v", "hevc_nvenc", "-cq", "25", "-preset", "p7", "-an",
            FACE_VIDEOS[face]
        ])

    # ----------------------
    # 3. Extract images
    # ----------------------
    for face, video in FACE_VIDEOS.items():
        out_dir = f"{OUTPUT_IMAGES_DIR}/{face}"
        run([
            "ffmpeg", "-i", video,
            "-vf", "fps=1",
            f"{out_dir}/{face}_%06d.png"
        ])

    print("Images generated at:")
    print(OUTPUT_IMAGES_DIR)

    # ----------------------
    # 4. Generate masks for 6 folders
    # ----------------------
    print("Loading YOLO model…")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8x-seg.pt")

    faces = ["front", "back", "up", "down", "left", "right"]

    for face in faces:
        print(f"[Mask] Processing {face}…")
        img_dir = Path(f"{OUTPUT_IMAGES_DIR}/{face}")
        mask_root = Path(f"{OUTPUT_MASKS_DIR}/{face}")

        images = collect_images(img_dir)
        if not images:
            continue

        for img_path in images:
            results = model.predict(
                source=str(img_path),
                classes=[0],
                conf=0.25,
                iou=0.45,
                device=device,
                verbose=False
            )
            r = results[0]
            H, W = r.orig_shape
            mask = np.full((H, W), 255, dtype=np.uint8)

            person_union = None
            if r.masks is not None and len(r.masks) > 0:
                m = r.masks.data
                m = (m > 0.5).cpu().numpy()
                m = resize_bool_mask_to(m, W, H)
                keep = [mk for mk in m if mk.sum() >= 5000]
                if keep:
                    person_union = np.any(np.stack(keep, axis=0), axis=0)

            if person_union is None:
                person_union = np.zeros((H, W), dtype=bool)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            person_union = cv2.dilate(person_union.astype(np.uint8), kernel) > 0

            mask[person_union] = 0

            out_path = make_mask_path(mask_root, img_dir, img_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), mask)

    print("All masks saved at:")
    print(OUTPUT_MASKS_DIR)

    # ----------------------
    # 5. Merge into final images/ and masks/
    # ----------------------
    FINAL_IMG = f"{DATASET_DIR}/images"
    FINAL_MASK = f"{DATASET_DIR}/masks"
    os.makedirs(FINAL_IMG, exist_ok=True)
    os.makedirs(FINAL_MASK, exist_ok=True)

    faces_list = ["front", "back", "up", "down", "left", "right"]

    # merge images
    for face in faces_list:
        src = Path(f"{OUTPUT_IMAGES_DIR}/{face}")
        for img in src.rglob("*.png"):
            dst = Path(FINAL_IMG) / f"{face}_{img.name}"
            dst.write_bytes(img.read_bytes())

    # merge masks
    for face in faces_list:
        src = Path(f"{OUTPUT_MASKS_DIR}/{face}")
        for m in src.rglob("*.png"):
            dst = Path(FINAL_MASK) / f"{face}_{m.name}"
            dst.write_bytes(m.read_bytes())

    print("Final merged folders ready for COLMAP:")
    print(FINAL_IMG)
    print(FINAL_MASK)


if __name__ == "__main__":
    main()
