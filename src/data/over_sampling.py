"""oversample_hole.py — Oversample minority classes in the sewage-yolo26 dataset.

For each target class, finds all training images that contain that class,
then generates augmented copies until the target count is reached.

Augmentations applied (mild, to avoid distribution shift):
  - Random horizontal flip
  - Random brightness / contrast jitter
  - Random rotation ±10 degrees
  - Random crop-and-resize (scale 0.85–1.0)

The script is IDEMPOTENT — it writes new files with a unique suffix
(_os_NNNN) so re-running never duplicates existing oversampled files.

Usage (run from repo root or SageMaker):
    python oversample_hole.py

    # Override dataset path:
    DATASET_ROOT=/path/to/sewage-yolo26 python oversample_hole.py

    # Override target counts:
    TARGET_COUNTS='{"Hole":200,"Buckling":200}' python oversample_hole.py
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger("oversample")

# ---------------------------------------------------------------------------
# Configuration — override via environment variables
# ---------------------------------------------------------------------------

# Path to the bbox dataset (sewage-yolo26).
DATASET_ROOT = Path(r"E:\cnn-assignment3-main\cnn-assignment3-main\src\data\sewage-yolo26")

# Only oversample the training split.
SPLIT = "train"

# Class names in the order defined in data.yaml.
CLASS_NAMES = [
    "Buckling",        # 0
    "Crack",           # 1
    "Debris",          # 2
    "Hole",            # 3
    "Joint offset",    # 4
    "Obstacle",        # 5
    "Utility intrusion",  # 6
]

# Target instance counts per class after oversampling.
# Only classes below their target will be oversampled.
_default_targets = {
    "Hole":     200,   # 31  → 200  (~6.5x)
    "Buckling": 200,   # 136 → 200  (~1.5x)
    "Obstacle": 200,   # 136 → 200  (~1.5x)
}
TARGET_COUNTS: dict[str, int] = json.loads(
    os.environ.get("TARGET_COUNTS", json.dumps(_default_targets))
)

# Random seed for reproducibility.
RANDOM_SEED = 42

# Image extensions recognised by the script.
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def _random_flip(img: np.ndarray, labels: list[list[float]]) -> tuple:
    """Horizontal flip with probability 0.5."""
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        new_labels = []
        for row in labels:
            cls, cx, cy, w, h = row
            new_labels.append([cls, 1.0 - cx, cy, w, h])
        return img, new_labels
    return img, labels


def _random_brightness_contrast(img: np.ndarray) -> np.ndarray:
    """Random brightness (±30) and contrast (0.8–1.2) jitter."""
    alpha = random.uniform(0.8, 1.2)   # contrast
    beta  = random.randint(-30, 30)    # brightness
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def _random_rotate(
    img: np.ndarray,
    labels: list[list[float]],
    max_deg: float = 10.0,
) -> tuple:
    """Small rotation — bbox is re-fitted as axis-aligned after rotation."""
    angle = random.uniform(-max_deg, max_deg)
    h, w  = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img_rot = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    new_labels = []
    for row in labels:
        cls, cx, cy, bw, bh = row
        # Convert to pixel corners, rotate, refit.
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        ones    = np.ones((4, 1), dtype=np.float32)
        corners_h = np.hstack([corners, ones])
        rotated   = (M @ corners_h.T).T
        rx1, ry1  = rotated[:, 0].min(), rotated[:, 1].min()
        rx2, ry2  = rotated[:, 0].max(), rotated[:, 1].max()
        # Clamp to image.
        rx1, ry1 = max(0.0, rx1), max(0.0, ry1)
        rx2, ry2 = min(float(w), rx2), min(float(h), ry2)
        ncx = ((rx1 + rx2) / 2) / w
        ncy = ((ry1 + ry2) / 2) / h
        nbw = (rx2 - rx1) / w
        nbh = (ry2 - ry1) / h
        if nbw > 0.01 and nbh > 0.01:
            new_labels.append([cls, ncx, ncy, nbw, nbh])
    return img_rot, new_labels


def _random_scale_crop(
    img: np.ndarray,
    labels: list[list[float]],
    scale_range: tuple = (0.85, 1.0),
) -> tuple:
    """Random crop to `scale` fraction then resize back to original dims."""
    scale = random.uniform(*scale_range)
    if scale >= 1.0:
        return img, labels
    h, w = img.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    top  = random.randint(0, h - new_h)
    left = random.randint(0, w - new_w)
    cropped = img[top:top + new_h, left:left + new_w]
    resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    new_labels = []
    for row in labels:
        cls, cx, cy, bw, bh = row
        # Map from full-image normalised coords → cropped region normalised coords.
        px1 = (cx - bw / 2) * w
        py1 = (cy - bh / 2) * h
        px2 = (cx + bw / 2) * w
        py2 = (cy + bh / 2) * h
        # Clip to crop window.
        cx1 = max(px1, left)
        cy1 = max(py1, top)
        cx2 = min(px2, left + new_w)
        cy2 = min(py2, top + new_h)
        if cx2 <= cx1 or cy2 <= cy1:
            continue  # box fully outside crop
        # Remap to resized image coords.
        ncx = ((cx1 + cx2) / 2 - left) / new_w
        ncy = ((cy1 + cy2) / 2 - top)  / new_h
        nbw = (cx2 - cx1) / new_w
        nbh = (cy2 - cy1) / new_h
        if nbw > 0.01 and nbh > 0.01:
            new_labels.append([cls, ncx, ncy, nbw, nbh])
    return resized, new_labels


def augment(img: np.ndarray, labels: list[list[float]]) -> tuple:
    """Apply all augmentations in sequence."""
    img, labels = _random_flip(img, labels)
    img         = _random_brightness_contrast(img)
    img, labels = _random_rotate(img, labels)
    img, labels = _random_scale_crop(img, labels)
    return img, labels

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _read_labels(label_path: Path) -> list[list[float]]:
    rows = []
    if not label_path.exists():
        return rows
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            rows.append([float(p) for p in parts[:5]])
    return rows


def _write_labels(label_path: Path, rows: list[list[float]]) -> None:
    lines = [f"{int(r[0])} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}" for r in rows]
    label_path.write_text("\n".join(lines), encoding="utf-8")


def _existing_oversample_count(img_dir: Path, stem_prefix: str) -> int:
    """Count already-generated oversampled files for a given original stem."""
    return sum(1 for p in img_dir.iterdir() if p.stem.startswith(stem_prefix + "_os_"))


def oversample(dataset_root: Path, split: str) -> None:
    img_dir = dataset_root / split / "images"
    lbl_dir = dataset_root / split / "labels"

    if not img_dir.exists():
        LOGGER.error("Image dir not found: %s", img_dir)
        return

    # ── 1. Build index: class_id → list of (img_path, label_path) ──────────
    class_to_images: dict[int, list[tuple[Path, Path]]] = defaultdict(list)
    all_images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)

    for img_path in all_images:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        rows = _read_labels(lbl_path)
        seen = set()
        for row in rows:
            cid = int(row[0])
            if cid not in seen:
                class_to_images[cid].append((img_path, lbl_path))
                seen.add(cid)

    # ── 2. Count current instances ──────────────────────────────────────────
    instance_counter: Counter = Counter()
    for img_path in all_images:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        for row in _read_labels(lbl_path):
            instance_counter[int(row[0])] += 1

    LOGGER.info("Current instance counts:")
    for cid, name in enumerate(CLASS_NAMES):
        LOGGER.info("  %-20s %4d", name, instance_counter.get(cid, 0))

    # ── 3. Oversample each target class ────────────────────────────────────
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    total_written = 0
    for class_name, target in TARGET_COUNTS.items():
        if class_name not in CLASS_NAMES:
            LOGGER.warning("Unknown class '%s' — skipping.", class_name)
            continue
        cid = CLASS_NAMES.index(class_name)
        current = instance_counter.get(cid, 0)
        needed  = max(0, target - current)

        if needed == 0:
            LOGGER.info("%-20s already at %d (target %d) — skipping.", class_name, current, target)
            continue

        source_images = class_to_images.get(cid, [])
        if not source_images:
            LOGGER.warning("No training images found for class '%s' — cannot oversample.", class_name)
            continue

        LOGGER.info(
            "Oversampling %-20s: %d → %d (need %d more, from %d source images)",
            class_name, current, target, needed, len(source_images),
        )

        generated = 0
        attempt   = 0
        while generated < needed:
            attempt += 1
            src_img_path, src_lbl_path = random.choice(source_images)
            img = cv2.imread(str(src_img_path))
            if img is None:
                LOGGER.warning("Could not read %s — skipping.", src_img_path)
                continue

            labels = _read_labels(src_lbl_path)
            if not labels:
                continue

            aug_img, aug_labels = augment(img, labels)

            # Check that the target class survived augmentation.
            if not any(int(r[0]) == cid for r in aug_labels):
                continue  # bbox was cropped out — retry

            # Write with unique suffix.
            suffix = f"_os_{generated:04d}"
            new_stem     = src_img_path.stem + suffix
            new_img_path = img_dir / (new_stem + src_img_path.suffix)
            new_lbl_path = lbl_dir / (new_stem + ".txt")

            cv2.imwrite(str(new_img_path), aug_img)
            _write_labels(new_lbl_path, aug_labels)
            generated += 1
            total_written += 1

            if attempt > needed * 10:
                LOGGER.warning(
                    "Too many retries for %s — stopping at %d/%d.", class_name, generated, needed
                )
                break

        LOGGER.info("  wrote %d new images for %s.", generated, class_name)

    # ── 4. Final counts ─────────────────────────────────────────────────────
    final_counter: Counter = Counter()
    for img_path in img_dir.iterdir():
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        for row in _read_labels(lbl_path):
            final_counter[int(row[0])] += 1

    LOGGER.info("")
    LOGGER.info("Final instance counts after oversampling:")
    LOGGER.info("  %-20s %6s %6s", "class", "before", "after")
    LOGGER.info("  " + "-" * 36)
    for cid, name in enumerate(CLASS_NAMES):
        before = instance_counter.get(cid, 0)
        after  = final_counter.get(cid, 0)
        delta  = after - before
        LOGGER.info("  %-20s %6d %6d  (+%d)", name, before, after, delta)
    LOGGER.info("  Total new files written: %d", total_written)


def main() -> None:
    # Auto-detect dataset root if not found at default path.
    global DATASET_ROOT
    if not DATASET_ROOT.exists():
        candidates = [
            Path("ass3/sewage-yolo26"),
            Path("../sewage-yolo26"),
            Path("../ass3/sewage-yolo26"),
            Path.cwd() / "sewage-yolo26",
        ]
        found = next((p for p in candidates if p.exists()), None)
        if found is None:
            raise FileNotFoundError(
                f"sewage-yolo26 not found. tried: {[str(p) for p in [DATASET_ROOT] + candidates]}"
            )
        DATASET_ROOT = found.resolve()

    LOGGER.info("Dataset root : %s", DATASET_ROOT)
    LOGGER.info("Split        : %s", SPLIT)
    LOGGER.info("Target counts: %s", TARGET_COUNTS)
    LOGGER.info("")

    oversample(DATASET_ROOT, SPLIT)
    LOGGER.info("Done.")


if __name__ == "__main__":
    main()