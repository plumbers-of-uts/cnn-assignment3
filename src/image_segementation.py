import os
import torch
from ultralytics import SAM
from PIL import Image
import cv2
import numpy as np

sam = SAM("sam2.1_b.pt")

BASE = r"E:\cnn-assignment3-main\cnn-assignment3-main\src\data\sewage-yolo26"


def read_yolo_labels(label_path, img_w=640, img_h=640):
    bboxes, classes = [], []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            bboxes.append([x1, y1, x2, y2])
            classes.append(cls)
    return torch.tensor(bboxes), classes


def run_single(img_path, label_path, save_path):
    bboxes, classes = read_yolo_labels(label_path)

    if len(bboxes) == 0:
        print(f"No annotations, skipping: {img_path}")
        return

    # SAM segmentation
    seg = sam(img_path, bboxes=bboxes)
    seg[0].save(save_path)
    print(f"Saved: {save_path}")



img_dir = os.path.join(BASE, "train", "images")
label_dir = os.path.join(BASE, "train", "labels")
out_dir = os.path.join(BASE, "output")
os.makedirs(out_dir, exist_ok=True)

img_files = sorted([f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))])

for fname in img_files[:3]:
    stem = os.path.splitext(fname)[0]
    img_path = os.path.join(img_dir, fname)
    label_path = os.path.join(label_dir, stem + ".txt")
    save_path = os.path.join(out_dir, stem + "_seg.jpg")

    run_single(img_path, label_path, save_path)