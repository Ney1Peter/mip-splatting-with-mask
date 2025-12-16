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
Preprocess pipeline for standard (perspective) video:
1. Extract frames at 1 FPS → images/
2. Generate YOLOv8 person masks (black=person, white=background)
3. Output two mask dirs:
   - masks/  : mask names EXACTLY match images (e.g., 000001.png)
   - mask/   : COLMAP-style double suffix (e.g., 000001.png.png)
Supports: .mp4, .mov, .avi, .mkv, etc. via ffmpeg.
"""

# ============================
# ✨ ABSOLUTE PATHS (LOCKED) ✨
# ============================
DATASET_DIR = "/home/zheng/mip-splatting-with-mask/data/garden2_mobile"

# 🔍 Auto-detect video file in DATASET_DIR
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
video_files = []
for ext in VIDEO_EXTENSIONS:
    video_files.extend(Path(DATASET_DIR).glob(f"*{ext}"))
    video_files.extend(Path(DATASET_DIR).glob(f"*{ext.upper()}"))

if not video_files:
    raise FileNotFoundError(f"No video file (.mp4, .mov, etc.) found in {DATASET_DIR}")

# If multiple videos, try to pick the one matching expected base name (optional)
# For example, prefer "tombstone_m.mp4" or "tombstone_m.mov"
PREFERRED_BASE = "tombstone_m"  # you can change this or remove preference
input_video = None
for vf in sorted(video_files):
    if vf.stem == PREFERRED_BASE:
        input_video = vf
        break

if input_video is None:
    # Just pick the first one if no preference match
    input_video = sorted(video_files)[0]

print(f"✅ Using video: {input_video}")

INPUT_VIDEO = str(input_video)

OUTPUT_IMAGES_DIR = f"{DATASET_DIR}/images"
OUTPUT_MASKS_STD_DIR = f"{DATASET_DIR}/masks"     # single suffix: 000001.png
OUTPUT_MASK_COLMAP_DIR = f"{DATASET_DIR}/mask"    # double suffix:  000001.png.png

# ============================
# Morphology parameters (tune as needed)
# ============================
MASK_MORPH_MODE = "dilate"          # "dilate" or "erode"
MASK_GROW_RADIUS_PX = 16
MASK_GROW_RADIUS_REL = 0.0          # overrides PX if >0
MASK_DILATE_ITERATIONS = 5
MASK_ERODE_ITERATIONS = 1
MASK_CLOSE_HOLES = False
MASK_CLOSE_RADIUS_PX = 3
MIN_INSTANCE_PIXELS = 5000

# ============================
# Helper functions
# ============================
def run(cmd):
    print("[Running]", " ".join(cmd))
    subprocess.run(cmd, check=True)

def collect_images(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return [p for p in sorted(root.rglob("*")) if p.suffix.lower() in exts]

def make_colmap_mask_path(img_path: Path) -> Path:
    """Convert image path to COLMAP double-suffix mask path."""
    return img_path.with_suffix(img_path.suffix + ".png")

def make_disk(radius_px: int):
    k = max(1, int(radius_px))
    size = 2 * k + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

def morph_mask(binary_bool: np.ndarray, H: int, W: int) -> np.ndarray:
    if MASK_GROW_RADIUS_REL > 0:
        r = int(round(min(H, W) * MASK_GROW_RADIUS_REL))
    else:
        r = int(MASK_GROW_RADIUS_PX)

    if r > 0:
        kernel = make_disk(r)
        if MASK_MORPH_MODE.lower() == "erode":
            binary_bool = cv2.erode(
                binary_bool.astype(np.uint8), kernel, iterations=MASK_ERODE_ITERATIONS
            ) > 0
        else:
            binary_bool = cv2.dilate(
                binary_bool.astype(np.uint8), kernel, iterations=MASK_DILATE_ITERATIONS
            ) > 0

    if MASK_CLOSE_HOLES:
        kr = max(1, int(MASK_CLOSE_RADIUS_PX))
        k_close = make_disk(kr)
        tmp = cv2.dilate(binary_bool.astype(np.uint8), k_close, iterations=1) > 0
        binary_bool = cv2.erode(tmp.astype(np.uint8), k_close, iterations=1) > 0

    return binary_bool

# ============================
# Main pipeline
# ============================
def main():
    # --- Prepare output dirs
    Path(OUTPUT_IMAGES_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_MASKS_STD_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_MASK_COLMAP_DIR).mkdir(parents=True, exist_ok=True)

    # --- Step 1: Extract images at 1 FPS (skip if already done)
    if not any(p.suffix.lower() in {".png", ".jpg", ".jpeg"} for p in Path(OUTPUT_IMAGES_DIR).rglob("*")):
        print("Extracting frames at 1 FPS...")
        run([
            "ffmpeg", "-i", INPUT_VIDEO,
            "-vf", "fps=1",
            str(Path(OUTPUT_IMAGES_DIR) / "%06d.png")
        ])
    else:
        print("[Skip] Images already exist in:", OUTPUT_IMAGES_DIR)

    # --- Step 2: Generate masks
    images = collect_images(Path(OUTPUT_IMAGES_DIR))
    if not images:
        print("[Error] No images found. Aborting.")
        return

    print("Loading YOLOv8 model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8x-seg.pt")

    for img_path in images:
        # Skip if both mask formats already exist
        std_mask = Path(OUTPUT_MASKS_STD_DIR) / img_path.name
        colmap_mask = Path(OUTPUT_MASK_COLMAP_DIR) / (img_path.name + ".png")
        if std_mask.exists() and colmap_mask.exists():
            continue

        print(f"Processing mask for {img_path.name}...")
        results = model.predict(
            source=str(img_path),
            classes=[0],      # person class
            conf=0.25,
            iou=0.45,
            device=device,
            verbose=False
        )
        r = results[0]
        H, W = r.orig_shape
        out_mask = np.full((H, W), 255, dtype=np.uint8)  # white background

        person_union = None
        if r.masks is not None and len(r.masks) > 0:
            masks = r.masks.data
            bool_masks = (masks > 0.5).cpu().numpy()
            # Resize to original image size
            resized_masks = []
            for m in bool_masks:
                mk_uint8 = (m.astype(np.uint8) * 255)
                mk_resized = cv2.resize(mk_uint8, (W, H), interpolation=cv2.INTER_NEAREST)
                resized_masks.append(mk_resized > 0)
            keep = [m for m in resized_masks if m.sum() >= MIN_INSTANCE_PIXELS]
            if keep:
                person_union = np.any(np.stack(keep, axis=0), axis=0)

        if person_union is None:
            person_union = np.zeros((H, W), dtype=bool)

        person_union = morph_mask(person_union, H=H, W=W)
        out_mask[person_union] = 0  # black for person

        # Save both formats
        cv2.imwrite(str(std_mask), out_mask)
        cv2.imwrite(str(colmap_mask), out_mask)

    print("✅ Done.")
    print("Images :", OUTPUT_IMAGES_DIR)
    print("Masks  :", OUTPUT_MASKS_STD_DIR, " (e.g., 000001.png)")
    print("Mask   :", OUTPUT_MASK_COLMAP_DIR, " (e.g., 000001.png.png)")

if __name__ == "__main__":
    main()