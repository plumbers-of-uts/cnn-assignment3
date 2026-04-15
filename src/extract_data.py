"""
Data Extraction Script

This script extracts images and labels from a YOLO-format dataset
and saves them as .pt files into separate folders:

- feature/ : image tensors
- label/   : label tensors

Image format: Tensor [C, H, W] (RGB, typically 640x640)
Label format: Tensor [N, 5] (class, xc, yc, w, h)
"""

import os
import torch
from torchvision import transforms
from PIL import Image

# ─── Path Configuration ───────────────────────────────────────────────
BASE_DIR    = r"F:\DNN_A3\D3\cnn-assignment3\src\data\sewage-yolo26\train"
IMG_DIR     = os.path.join(BASE_DIR, "images")
LABEL_DIR   = os.path.join(BASE_DIR, "labels")
FEATURE_DIR = os.path.join(BASE_DIR, "feature")
LABEL_OUT   = os.path.join(BASE_DIR, "label")

os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(LABEL_OUT,   exist_ok=True)

# ─── Image Preprocessing ──────────────────────────────────────────────
# Converts PIL image → Float Tensor [C, H, W] normalized to [0,1]
to_tensor = transforms.ToTensor()


def load_image(img_path: str) -> torch.Tensor:
    """
    Load an image and convert it to an RGB tensor.

    Returns:
        Tensor of shape [3, H, W], dtype=float32, values in [0.0, 1.0]
    """
    img = Image.open(img_path).convert("RGB")
    return to_tensor(img)


def load_label(txt_path: str) -> torch.Tensor:
    """
    Load YOLO-format label file.

    Each line: class xc yc w h

    Returns:
        Tensor of shape [N, 5], dtype=float32
        If file is empty or missing, returns an empty tensor [0, 5]
    """
    rows = []
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    vals = list(map(float, line.split()))
                    rows.append(vals)

    if rows:
        return torch.tensor(rows, dtype=torch.float32)
    else:
        return torch.zeros((0, 5), dtype=torch.float32)


# ─── Main Extraction Logic ────────────────────────────────────────────
def extract_all():
    img_files = sorted(
        [f for f in os.listdir(IMG_DIR) if f.lower().endswith(".jpg")]
    )

    print(f"Found {len(img_files)} images. Starting extraction...\n")

    for img_name in img_files:
        stem = os.path.splitext(img_name)[0]

        img_path   = os.path.join(IMG_DIR,   img_name)
        label_path = os.path.join(LABEL_DIR, stem + ".txt")
        feat_out   = os.path.join(FEATURE_DIR, stem + ".pt")
        lbl_out    = os.path.join(LABEL_OUT,   stem + ".pt")

        # Extract image tensor
        feature_tensor = load_image(img_path)
        torch.save(feature_tensor, feat_out)

        # Extract label tensor
        label_tensor = load_label(label_path)
        torch.save(label_tensor, lbl_out)

    print("Extraction completed!")
    print(f"  feature/ -> {FEATURE_DIR}")
    print(f"  label/   -> {LABEL_OUT}\n")


# ─── Test: Inspect First Three Samples ────────────────────────────────
def test_first_three():
    feat_files = sorted(
        [f for f in os.listdir(FEATURE_DIR) if f.endswith(".pt")]
    )[:3]

    print("=" * 60)
    print("Test: Inspect first three samples (INPUT and LABEL)")
    print("=" * 60)

    for i, fname in enumerate(feat_files, 1):
        stem = os.path.splitext(fname)[0]

        feat  = torch.load(os.path.join(FEATURE_DIR, fname))
        label = torch.load(os.path.join(LABEL_OUT,   stem + ".pt"))

        C, H, W = feat.shape
        N       = label.shape[0]

        # ── INPUT Preview ─────────────────────────────────────────
        # Display first 3 pixels from top-left corner
        input_preview = []
        for row in range(min(1, H)):
            for col in range(min(3, W)):
                r = feat[0, row, col].item()
                g = feat[1, row, col].item()
                b = feat[2, row, col].item()
                input_preview.append(
                    f"[RGB=({r:.3f},{g:.3f},{b:.3f}), W={col}, H={row}]"
                )

        # ── LABEL Preview ─────────────────────────────────────────
        label_strs = []
        for j in range(N):
            cls, xc, yc, w, h = label[j].tolist()
            label_strs.append(
                f"[class={int(cls)}, xc={xc:.4f}, yc={yc:.4f}, w={w:.4f}, h={h:.4f}]"
            )

        print(f"\n── Sample {i}: {stem} ──")
        print(f"  Tensor shape: feature={list(feat.shape)} (C×H×W)")
        print(f"  dtype:        {feat.dtype}")
        print(f"  Pixel range:  [{feat.min():.4f}, {feat.max():.4f}]")

        print(f"\n  INPUT (top-left 3 pixels preview):")
        for p in input_preview:
            print(f"    {p}")
        print(f"  … total pixels: {W * H}")

        print(f"\n  LABEL ({N} bounding boxes):")
        if label_strs:
            for ls in label_strs:
                print(f"    {ls}")
        else:
            print("    (no objects)")

    print("\n" + "=" * 60)


# ─── Entry Point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    extract_all()
    test_first_three()
