"""generate_seg_labels.py — Convert YOLO bbox labels to segmentation polygons via SAM.

For each image in train/valid/test of the Roboflow sewage-yolo26 dataset, this
script:

1. Reads the existing YOLO bbox labels.
2. Calls Ultralytics SAM (sam2.1_b.pt) with each bbox as a prompt.
3. Extracts the largest contour from each binary mask via cv2.findContours.
4. Simplifies the polygon with Douglas-Peucker.
5. Writes YOLO segmentation labels: ``cls x1 y1 x2 y2 ... xn yn`` with normalised
   coordinates, one detection per line.
6. Symlinks the original image into the seg dataset directory so disk usage
   only grows by the label files.

The output dataset (``src/data/sewage-yolo26-seg``) is suitable for training a
``yolo26m-seg`` model on the same 7 classes.

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python src/generate_seg_labels.py
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import SAM  # type: ignore[attr-defined]

LOGGER = logging.getLogger("seg-label-gen")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SOURCE_DATASET = Path("src/data/sewage-yolo26")
OUTPUT_DATASET = Path("src/data/sewage-yolo26-seg")
SAM_CHECKPOINT = "sam2.1_b.pt"
SPLITS: tuple[str, ...] = ("train", "valid", "test")
IMAGE_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png")

# Image dimensions of the Roboflow export (verified — all images are 640x640).
IMG_WIDTH = 640
IMG_HEIGHT = 640

# Polygon simplification — larger epsilon = fewer points, lower fidelity.
POLY_EPSILON_PX = 2.0
# Polygons with fewer points than this are skipped (degenerate).
MIN_POLYGON_POINTS = 3
# Skip masks whose area is less than this fraction of the bbox area
# (filters SAM hallucinations near zero).
MIN_MASK_AREA_RATIO = 0.05

# Minimum fields per YOLO bbox label row: class + 4 bbox values.
YOLO_BBOX_MIN_FIELDS = 5
# Mask tensor must be a 2D binary array.
MASK_NDIM = 2


@dataclass
class GenerationStats:
    """Aggregated counters for end-of-run reporting."""

    images_processed: int = 0
    images_with_no_labels: int = 0
    bboxes_total: int = 0
    masks_kept: int = 0
    masks_dropped_empty: int = 0
    masks_dropped_too_small: int = 0
    masks_dropped_degenerate: int = 0
    class_counter: Counter[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Initialise the class counter lazily so dataclass default stays mutable-safe."""
        if self.class_counter is None:
            self.class_counter = Counter()


def _read_yolo_bbox_labels(
    label_path: Path, img_w: int = IMG_WIDTH, img_h: int = IMG_HEIGHT
) -> tuple[list[list[float]], list[int]]:
    """Parse a YOLO bbox label file into pixel-space xyxy + class id lists."""
    bboxes: list[list[float]] = []
    classes: list[int] = []
    if not label_path.exists():
        return bboxes, classes

    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) < YOLO_BBOX_MIN_FIELDS:
            continue
        cls = int(parts[0])
        cx, cy, w, h = (float(v) for v in parts[1:5])
        # YOLO normalised (cx, cy, w, h) → absolute (x1, y1, x2, y2).
        x1 = (cx - w / 2.0) * img_w
        y1 = (cy - h / 2.0) * img_h
        x2 = (cx + w / 2.0) * img_w
        y2 = (cy + h / 2.0) * img_h
        bboxes.append([x1, y1, x2, y2])
        classes.append(cls)
    return bboxes, classes


def _mask_to_polygon(
    mask: np.ndarray[Any, Any],
    bbox_area_px: float,
) -> np.ndarray[Any, Any] | None:
    """Convert a binary mask to a single normalised polygon (largest contour)."""
    if mask.ndim != MASK_NDIM:
        msg = f"Expected 2D mask, got shape {mask.shape}"
        raise ValueError(msg)

    mask_uint8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    largest = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(largest))
    # Filter SAM hallucinations whose mask is much smaller than the bbox prompt.
    if bbox_area_px > 0.0 and (contour_area / bbox_area_px) < MIN_MASK_AREA_RATIO:
        return None

    simplified = cv2.approxPolyDP(largest, POLY_EPSILON_PX, closed=True)
    points = simplified.reshape(-1, 2)
    if len(points) < MIN_POLYGON_POINTS:
        return None
    return points


def _resolve_sam_masks(
    sam: SAM,
    image_path: Path,
    bboxes: list[list[float]],
) -> list[np.ndarray[Any, Any]]:
    """Run SAM with the bbox prompts and return a list of binary masks."""
    if not bboxes:
        return []

    bbox_tensor = torch.tensor(bboxes, dtype=torch.float32)
    result = sam(str(image_path), bboxes=bbox_tensor, verbose=False)[0]

    if result.masks is None or len(result.masks) == 0:
        return []

    # masks.data shape: (num_boxes, H, W) on the model device.
    return [m.detach().cpu().numpy() for m in result.masks.data]


def _format_polygon_line(cls_id: int, polygon: np.ndarray[Any, Any]) -> str:
    """Serialise a polygon as a YOLO seg label row."""
    coords: list[str] = []
    for x_px, y_px in polygon:
        x_norm = float(x_px) / IMG_WIDTH
        y_norm = float(y_px) / IMG_HEIGHT
        # Clamp to [0, 1] to defend against off-by-one outliers near the border.
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))
        coords.append(f"{x_norm:.6f} {y_norm:.6f}")
    return f"{cls_id} " + " ".join(coords)


def _ensure_image_symlink(src_image: Path, dst_image: Path) -> None:
    """Symlink the source image into the destination dir (idempotent)."""
    dst_image.parent.mkdir(parents=True, exist_ok=True)
    if dst_image.exists() or dst_image.is_symlink():
        return
    dst_image.symlink_to(src_image.resolve())


def _process_split(
    split: str,
    sam: SAM,
    stats: GenerationStats,
    *,
    force: bool = False,
) -> None:
    """Generate seg labels for one dataset split."""
    src_images_dir = SOURCE_DATASET / split / "images"
    src_labels_dir = SOURCE_DATASET / split / "labels"

    if not src_images_dir.exists():
        LOGGER.warning("Split %s skipped: missing %s", split, src_images_dir)
        return

    dst_images_dir = OUTPUT_DATASET / split / "images"
    dst_labels_dir = OUTPUT_DATASET / split / "labels"
    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        p for p in src_images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    LOGGER.info("Split %s: %d images", split, len(image_files))

    skipped_existing = 0
    for image_path in image_files:
        stats.images_processed += 1
        dst_label_path = dst_labels_dir / f"{image_path.stem}.txt"
        dst_image_path = dst_images_dir / image_path.name
        _ensure_image_symlink(image_path, dst_image_path)

        # Resume mode: skip the (expensive) SAM call if the seg label already exists.
        if not force and dst_label_path.exists():
            skipped_existing += 1
            continue

        label_path = src_labels_dir / f"{image_path.stem}.txt"
        bboxes, classes = _read_yolo_bbox_labels(label_path)

        if not bboxes:
            stats.images_with_no_labels += 1
            # Empty seg label keeps dataset counts consistent with the bbox version.
            dst_label_path.write_text("", encoding="utf-8")
            continue

        stats.bboxes_total += len(bboxes)

        try:
            masks = _resolve_sam_masks(sam=sam, image_path=image_path, bboxes=bboxes)
        except Exception as exc:  # noqa: BLE001 — SAM should never abort the run.
            LOGGER.warning("SAM failed on %s (%s); skipping image.", image_path.name, exc)
            continue

        lines: list[str] = []
        for cls, bbox, mask in zip(classes, bboxes, masks, strict=False):
            bbox_area_px = max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            polygon = _mask_to_polygon(mask=mask, bbox_area_px=bbox_area_px)
            if polygon is None:
                if mask is None or mask.sum() == 0:
                    stats.masks_dropped_empty += 1
                elif bbox_area_px <= 0.0:
                    stats.masks_dropped_degenerate += 1
                else:
                    stats.masks_dropped_too_small += 1
                continue
            lines.append(_format_polygon_line(cls, polygon))
            stats.class_counter[cls] += 1
            stats.masks_kept += 1

        # Always symlink the image so Ultralytics can find it during training,
        # even when SAM produces no usable masks for it.
        _ensure_image_symlink(image_path, dst_images_dir / image_path.name)
        (dst_labels_dir / f"{image_path.stem}.txt").write_text(
            "\n".join(lines), encoding="utf-8"
        )

    LOGGER.info(
        "Split %s done | resumed=%d bboxes=%d masks_kept=%d empty=%d too_small=%d degenerate=%d",
        split,
        skipped_existing,
        stats.bboxes_total,
        stats.masks_kept,
        stats.masks_dropped_empty,
        stats.masks_dropped_too_small,
        stats.masks_dropped_degenerate,
    )


def _write_dataset_yaml() -> Path:
    """Emit data.yaml for Ultralytics seg training."""
    yaml_path = OUTPUT_DATASET / "data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    contents = (
        "path: ../sewage-yolo26-seg\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "task: segment\n"
        "nc: 7\n"
        "names: ['Buckling', 'Crack', 'Debris', 'Hole', 'Joint offset', "
        "'Obstacle', 'Utility intrusion']\n"
    )
    yaml_path.write_text(contents, encoding="utf-8")
    LOGGER.info("Wrote dataset yaml: %s", yaml_path)
    return yaml_path


def _verify_source_dataset() -> None:
    """Fail loud when the bbox dataset is not where the script expects it."""
    if not SOURCE_DATASET.exists():
        msg = (
            f"Source dataset missing: {SOURCE_DATASET}. "
            "Run from repo root and ensure Roboflow export is in place."
        )
        raise FileNotFoundError(msg)

    for split in SPLITS:
        if not (SOURCE_DATASET / split / "images").exists():
            LOGGER.warning("Source split %s images dir is missing; skipping.", split)


def _load_sam_model() -> SAM:
    """Instantiate Ultralytics SAM with the configured checkpoint."""
    LOGGER.info("Loading SAM checkpoint: %s", SAM_CHECKPOINT)
    return SAM(SAM_CHECKPOINT)


def main() -> None:
    """Entry point — generate seg labels for the entire dataset.

    Resumes by default: images whose seg label + image symlink already exist
    are skipped. Set the env var ``FORCE_REGENERATE=1`` to redo everything.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    _verify_source_dataset()

    force = os.environ.get("FORCE_REGENERATE", "").lower() in {"1", "true", "yes"}
    if OUTPUT_DATASET.exists():
        if force:
            LOGGER.warning("FORCE_REGENERATE=1 — every image will be re-processed.")
        else:
            LOGGER.info("Resuming: existing seg labels in %s will be skipped.", OUTPUT_DATASET)

    sam = _load_sam_model()

    stats = GenerationStats()
    for split in SPLITS:
        _process_split(split=split, sam=sam, stats=stats, force=force)

    yaml_path = _write_dataset_yaml()

    LOGGER.info(
        "Generation complete | images=%d no_labels=%d bboxes=%d masks_kept=%d",
        stats.images_processed,
        stats.images_with_no_labels,
        stats.bboxes_total,
        stats.masks_kept,
    )
    LOGGER.info("Per-class polygon counts: %s", dict(stats.class_counter))
    LOGGER.info("data.yaml ready: %s", yaml_path)

    # Clean up any zero-byte image symlink errors that might surface later.
    for split in SPLITS:
        broken = [
            p
            for p in (OUTPUT_DATASET / split / "images").glob("*")
            if p.is_symlink() and not p.exists()
        ]
        if broken:
            LOGGER.warning("Found %d broken symlinks in %s; removing.", len(broken), split)
            for p in broken:
                p.unlink()


if __name__ == "__main__":
    main()
