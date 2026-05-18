"""refine_seg_labels.py — Post-process SAM-generated polygons with class-specific fixes.

The SAM 2.1_b pseudo-polygons emitted by ``generate_seg_labels.py`` are noisy
on three failure modes that this script targets:

1. **Hole** masks bleed into surrounding concrete — SAM was prompted with a
   square bbox so it tends to grow the mask to the bbox extent. Replace each
   Hole polygon with an ellipse inscribed in the original bbox so the mask
   shape is consistent and shape-prior-friendly.

2. **Crack** masks come out as fat blobs instead of thin elongated paths,
   because SAM's bbox prompt does not encode the centerline. Approximate the
   medial axis via OpenCV morphological skeleton, then re-dilate to a fixed
   thickness so every Crack label has the same line-width prior.

3. **Buckling** and **Joint offset** masks include both hallucinated nubs
   (mask area « bbox area) and "ate the whole pipe" cases (mask area ≈ bbox
   area). Drop polygons whose area is outside ``[10%, 85%]`` of their source
   bbox area.

Run:
    uv run python src/refine_seg_labels.py
    # → writes refined labels to src/data/sewage-yolo26-seg-refined/

The notebook's dataset locator (Cell 6 of sagemaker_seg_train.ipynb) will
prefer ``sewage-yolo26-seg-refined`` over ``sewage-yolo26-seg`` if both
exist — add the refined path to its ``CANDIDATES`` list to switch.
"""

from __future__ import annotations

import logging
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
UInt8Array = NDArray[np.uint8]
Bbox = tuple[float, float, float, float]

LOGGER = logging.getLogger("refine-seg-labels")

# Class IDs — must match data.yaml ``names`` order.
CLS_BUCKLING = 0
CLS_CRACK = 1
CLS_HOLE = 3
CLS_JOINT_OFFSET = 4

SEG_DATASET = Path("src/data/sewage-yolo26-seg")
BBOX_DATASET = Path("src/data/sewage-yolo26")
OUTPUT_DATASET = Path("src/data/sewage-yolo26-seg-refined")
SPLITS: tuple[str, ...] = ("train", "valid", "test")
IMAGE_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png")

# Roboflow export is always 640x640 (verified). Hard-coding avoids loading
# every image just to read its shape.
IMG_W = 640
IMG_H = 640

ELLIPSE_POINTS = 24
CRACK_THICKNESS_PX = 4
MIN_AREA_RATIO = 0.10
MAX_AREA_RATIO = 0.85
MIN_CRACK_SKELETON_PX = 5
POLY_EPSILON_PX = 2.0
MIN_POLY_POINTS = 3
MAX_THINNING_ITER = 64

YOLO_BBOX_FIELDS = 5


@dataclass
class RefineStats:
    """End-of-run counters per refinement bucket."""

    rows_in: int = 0
    rows_out: int = 0
    hole_replaced: int = 0
    hole_no_bbox: int = 0
    crack_thinned: int = 0
    crack_dropped_short: int = 0
    crack_kept_raw: int = 0
    area_filter_dropped: int = 0
    degenerate_dropped: int = 0
    kept_unchanged: int = 0
    class_counter: Counter[int] = field(default_factory=Counter)


def _read_polygon_label(path: Path) -> list[tuple[int, FloatArray]]:
    """Parse a YOLO seg label file into [(cls, polygon_norm_xy)] tuples."""
    rows: list[tuple[int, FloatArray]] = []
    if not path.exists():
        return rows
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) < 1 + 2 * MIN_POLY_POINTS:
            continue
        cls = int(parts[0])
        coords = np.array(parts[1:], dtype=np.float64)
        # Drop odd trailing values rather than crash on malformed rows.
        coords = coords[: (coords.size // 2) * 2].reshape(-1, 2)
        if coords.shape[0] < MIN_POLY_POINTS:
            continue
        rows.append((cls, coords))
    return rows


def _read_bbox_index(path: Path) -> dict[int, list[Bbox]]:
    """Return YOLO bboxes (cx,cy,w,h) grouped by class id."""
    index: dict[int, list[Bbox]] = {}
    if not path.exists():
        return index
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) < YOLO_BBOX_FIELDS:
            continue
        cls = int(parts[0])
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        index.setdefault(cls, []).append((cx, cy, bw, bh))
    return index


def _polygon_centroid(poly_norm: FloatArray) -> tuple[float, float]:
    return float(poly_norm[:, 0].mean()), float(poly_norm[:, 1].mean())


def _match_bbox(poly_norm: FloatArray, candidates: list[Bbox]) -> Bbox | None:
    """Pick the bbox whose centre is closest to the polygon centroid."""
    if not candidates:
        return None
    cx, cy = _polygon_centroid(poly_norm)
    return min(candidates, key=lambda b: (b[0] - cx) ** 2 + (b[1] - cy) ** 2)


def _polygon_to_mask(poly_norm: FloatArray, h: int = IMG_H, w: int = IMG_W) -> UInt8Array:
    pts: NDArray[np.int32] = (poly_norm * np.array([w, h])).astype(np.int32)
    mask: UInt8Array = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], (1,))
    return mask


def _mask_to_polygon_norm(mask: UInt8Array) -> FloatArray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    simplified = cv2.approxPolyDP(largest, POLY_EPSILON_PX, closed=True).reshape(-1, 2)
    if simplified.shape[0] < MIN_POLY_POINTS:
        return None
    h, w = mask.shape
    result: FloatArray = simplified.astype(np.float64) / np.array([w, h], dtype=np.float64)
    return result


def _ellipse_from_bbox(bbox: Bbox, n_points: int = ELLIPSE_POINTS) -> FloatArray:
    """YOLO bbox (cx,cy,w,h) → normalized ellipse polygon inscribed in the bbox."""
    cx, cy, bw, bh = bbox
    rx, ry = bw / 2.0, bh / 2.0
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    xs = np.clip(cx + rx * np.cos(theta), 0.0, 1.0)
    ys = np.clip(cy + ry * np.sin(theta), 0.0, 1.0)
    result: FloatArray = np.stack([xs, ys], axis=1)
    return result


def _morphological_skeleton(mask: UInt8Array, max_iter: int = MAX_THINNING_ITER) -> UInt8Array:
    """Approximate medial axis via repeated open-residue (classic CV recipe)."""
    skeleton: UInt8Array = np.zeros_like(mask)
    temp = mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(max_iter):
        opened = cv2.morphologyEx(temp, cv2.MORPH_OPEN, kernel)
        residue: UInt8Array = cv2.subtract(temp, opened).astype(np.uint8)
        skeleton = cv2.bitwise_or(skeleton, residue).astype(np.uint8)
        temp = cv2.erode(temp, kernel).astype(np.uint8)
        if cv2.countNonZero(temp) == 0:
            break
    return skeleton


def _thin_crack(poly_norm: FloatArray, thickness: int = CRACK_THICKNESS_PX) -> FloatArray | None:
    """SAM-fat crack polygon → ``thickness``-px-thick polygon along the medial axis."""
    mask = _polygon_to_mask(poly_norm)
    if mask.sum() < MIN_CRACK_SKELETON_PX:
        return None
    skeleton = _morphological_skeleton(mask)
    if cv2.countNonZero(skeleton) < MIN_CRACK_SKELETON_PX:
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
    thickened: UInt8Array = cv2.dilate(skeleton, kernel).astype(np.uint8)
    return _mask_to_polygon_norm(thickened)


def _bbox_area_norm(bbox: Bbox | None) -> float:
    if bbox is None:
        return 0.0
    return bbox[2] * bbox[3]


def _polygon_area_norm(poly_norm: FloatArray) -> float:
    """Shoelace formula on normalized coords → area in [0,1] units."""
    xs = poly_norm[:, 0]
    ys = poly_norm[:, 1]
    return float(0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))))


def _within_area_band(poly_norm: FloatArray, bbox: Bbox | None) -> bool:
    ba = _bbox_area_norm(bbox)
    if ba <= 0.0:
        return True
    ratio = _polygon_area_norm(poly_norm) / ba
    return MIN_AREA_RATIO <= ratio <= MAX_AREA_RATIO


def _serialise(cls: int, poly_norm: FloatArray) -> str:
    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly_norm)
    return f"{cls} {coords}"


def _refine_row(
    cls: int,
    poly_norm: FloatArray,
    bboxes_by_cls: dict[int, list[Bbox]],
    stats: RefineStats,
) -> tuple[int, FloatArray] | None:
    """Apply the class-specific rule and return the refined (cls, polygon) or None to drop."""
    bbox = _match_bbox(poly_norm, bboxes_by_cls.get(cls, []))

    if cls == CLS_HOLE:
        if bbox is None:
            stats.hole_no_bbox += 1
            return None
        stats.hole_replaced += 1
        return cls, _ellipse_from_bbox(bbox)

    if cls == CLS_CRACK:
        thinned = _thin_crack(poly_norm)
        if thinned is None or len(thinned) < MIN_POLY_POINTS:
            # Skeleton too short — keep the raw polygon rather than drop the label.
            stats.crack_kept_raw += 1
            return cls, poly_norm
        stats.crack_thinned += 1
        return cls, thinned

    if cls in (CLS_BUCKLING, CLS_JOINT_OFFSET) and not _within_area_band(poly_norm, bbox):
        stats.area_filter_dropped += 1
        return None

    stats.kept_unchanged += 1
    return cls, poly_norm


def _process_split(split: str, stats: RefineStats) -> None:
    src_labels_dir = SEG_DATASET / split / "labels"
    src_images_dir = SEG_DATASET / split / "images"
    bbox_labels_dir = BBOX_DATASET / split / "labels"
    dst_labels_dir = OUTPUT_DATASET / split / "labels"
    dst_images_dir = OUTPUT_DATASET / split / "images"

    if not src_labels_dir.exists():
        LOGGER.warning("Split %s: missing %s — skipping.", split, src_labels_dir)
        return

    dst_labels_dir.mkdir(parents=True, exist_ok=True)
    dst_images_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(src_labels_dir.glob("*.txt"))
    LOGGER.info("Split %s: %d label files", split, len(label_files))

    for label_path in label_files:
        rows = _read_polygon_label(label_path)
        stats.rows_in += len(rows)
        bboxes_by_cls = _read_bbox_index(bbox_labels_dir / label_path.name)

        kept_lines: list[str] = []
        for cls, poly_norm in rows:
            if poly_norm.shape[0] < MIN_POLY_POINTS:
                stats.degenerate_dropped += 1
                continue
            result = _refine_row(cls, poly_norm, bboxes_by_cls, stats)
            if result is None:
                if cls == CLS_CRACK:
                    stats.crack_dropped_short += 1
                continue
            new_cls, new_poly = result
            kept_lines.append(_serialise(new_cls, new_poly))
            stats.class_counter[new_cls] += 1
            stats.rows_out += 1

        (dst_labels_dir / label_path.name).write_text(
            "\n".join(kept_lines), encoding="utf-8"
        )

        # Mirror the image as a symlink so Ultralytics can find it. Use absolute
        # source so the link survives `cwd` changes during training.
        for ext in IMAGE_EXTS:
            src_img = src_images_dir / f"{label_path.stem}{ext}"
            if not src_img.exists():
                continue
            dst_img = dst_images_dir / src_img.name
            if dst_img.is_symlink() or dst_img.exists():
                dst_img.unlink()
            dst_img.symlink_to(src_img.resolve())
            break

    LOGGER.info(
        "Split %s done | rows_in=%d rows_out=%d hole_replaced=%d crack_thinned=%d "
        "area_dropped=%d crack_kept_raw=%d",
        split,
        stats.rows_in,
        stats.rows_out,
        stats.hole_replaced,
        stats.crack_thinned,
        stats.area_filter_dropped,
        stats.crack_kept_raw,
    )


def _write_dataset_yaml() -> Path:
    yaml_path = OUTPUT_DATASET / "data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        "path: ../sewage-yolo26-seg-refined\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "task: segment\n"
        "nc: 7\n"
        "names: ['Buckling', 'Crack', 'Debris', 'Hole', 'Joint offset', "
        "'Obstacle', 'Utility intrusion']\n",
        encoding="utf-8",
    )
    return yaml_path


def main() -> None:
    """Refine ``sewage-yolo26-seg/`` → ``sewage-yolo26-seg-refined/``."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not SEG_DATASET.exists():
        msg = f"Source seg dataset missing: {SEG_DATASET}. Run generate_seg_labels.py first."
        raise FileNotFoundError(msg)

    # Replace the output dir cleanly so re-runs are reproducible.
    if OUTPUT_DATASET.exists():
        LOGGER.warning("Removing existing %s before refinement.", OUTPUT_DATASET)
        shutil.rmtree(OUTPUT_DATASET)

    stats = RefineStats()
    for split in SPLITS:
        _process_split(split, stats)

    yaml_path = _write_dataset_yaml()

    LOGGER.info(
        "Refine complete | rows_in=%d rows_out=%d kept_unchanged=%d "
        "hole_replaced=%d crack_thinned=%d crack_kept_raw=%d "
        "area_dropped=%d hole_no_bbox=%d",
        stats.rows_in,
        stats.rows_out,
        stats.kept_unchanged,
        stats.hole_replaced,
        stats.crack_thinned,
        stats.crack_kept_raw,
        stats.area_filter_dropped,
        stats.hole_no_bbox,
    )
    LOGGER.info("Per-class polygons after refine: %s", dict(stats.class_counter))
    LOGGER.info("data.yaml: %s", yaml_path)
    LOGGER.info(
        "Point sagemaker_seg_train.ipynb Cell 6 at this dir (add %s to CANDIDATES).",
        OUTPUT_DATASET,
    )


if __name__ == "__main__":
    main()
