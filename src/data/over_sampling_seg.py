r"""over_sampling_seg.py — Polygon-aware oversample for sewage-yolo26-seg.

Mirrors over_sampling.py but for instance-segmentation labels (polygon).
Augments minority classes by generating rotated / flipped / cropped copies
of source images, with polygon vertices transformed alongside the image
via a mask-raster round-trip (cv2.fillPoly -> warpAffine/crop ->
cv2.findContours) so rotation and crop work correctly even when polygons
get split or partially clipped.

Source : sewage-yolo26-seg/                  (686 imgs, polygon labels)
Output : sewage-yolo26-seg-oversample/
         |-- train/  (full copy of 686 + ~230 new oversampled)
         |-- valid/  (full copy -- eval consistency with v1/v2)
         |-- test/   (full copy -- no leakage)
         |-- data.yaml

Augmentations (polygon-aware, same recipe as over_sampling.py):
  - Random horizontal flip
  - Random brightness / contrast jitter (image-only)
  - Random rotation +/-10 degrees
  - Random scale-crop (0.85 - 1.0)

Idempotent: writes with _os_NNNN suffix, re-runs skip existing files.

Usage:
    python src/data/over_sampling_seg.py

    SRC_DATASET=/abs/path/to/sewage-yolo26-seg \\
    DST_DATASET=/abs/path/to/sewage-yolo26-seg-oversample \\
        python src/data/over_sampling_seg.py

    TARGET_COUNTS='{"Buckling":250,"Joint offset":250}' \\
        python src/data/over_sampling_seg.py
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

NDArr = np.ndarray[Any, Any]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger("oversample_seg")

# ---------------------------------------------------------------------------
# Configuration -- override via env vars
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]

SRC_DATASET = Path(
    os.environ.get("SRC_DATASET", str(_REPO_ROOT / "src/data/sewage-yolo26-seg")),
)
DST_DATASET = Path(
    os.environ.get("DST_DATASET", str(_REPO_ROOT / "src/data/sewage-yolo26-seg-oversample")),
)

CLASS_NAMES: list[str] = [
    "Buckling",          # 0
    "Crack",             # 1
    "Debris",            # 2
    "Hole",              # 3
    "Joint offset",      # 4
    "Obstacle",          # 5
    "Utility intrusion", # 6
]

# Targets mirror over_sampling.py (bbox dataset) so the segmentation oversample matches the
# RT-DETR training distribution -- this isolates "model architecture" as the only variable
# when comparing v3 (yolo26m-seg) against the RT-DETR + SAM line.
#   bbox  + over_sampling.py     -> Buckling 237, Hole 233, Obstacle 217
#   poly  + over_sampling_seg.py -> aim for the same band (exact match impossible because SAM
#                                   polygon conversion drops some instances)
_default_targets: dict[str, int] = {
    "Hole": 200,
    "Buckling": 200,
    "Obstacle": 200,
}
TARGET_COUNTS: dict[str, int] = json.loads(
    os.environ.get("TARGET_COUNTS", json.dumps(_default_targets)),
)

RANDOM_SEED = 42
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Polygon validity thresholds after augmentation.
MIN_VERTICES = 3                # Ultralytics needs >= 3 polygon vertices (>= 6 coords)
MIN_AREA_FRAC = 1e-4            # 0.01% of image area; smaller polygons are dropped
APPROX_EPS_PX = 0.5             # cv2.approxPolyDP epsilon in pixels
FLIP_PROB = 0.5                 # horizontal flip probability

# ---------------------------------------------------------------------------
# Polygon I/O (normalized [0,1] <-> pixel coords)
# ---------------------------------------------------------------------------


def _read_polygons(
    label_path: Path,
    img_w: int,
    img_h: int,
) -> list[tuple[int, NDArr]]:
    """Read polygon labels, returning [(class_id, [N,2] pixel-coord vertices), ...]."""
    polys: list[tuple[int, NDArr]] = []
    if not label_path.exists():
        return polys
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 1 + MIN_VERTICES * 2:
            continue
        try:
            cls = int(parts[0])
            coords = [float(p) for p in parts[1:]]
        except ValueError:
            continue
        if len(coords) % 2 != 0:
            continue
        pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= img_w
        pts[:, 1] *= img_h
        polys.append((cls, pts))
    return polys


def _write_polygons(
    label_path: Path,
    polys: list[tuple[int, NDArr]],
    img_w: int,
    img_h: int,
) -> None:
    lines: list[str] = []
    for cls, pts in polys:
        pts_norm = pts.astype(np.float64).copy()
        pts_norm[:, 0] /= img_w
        pts_norm[:, 1] /= img_h
        pts_norm = np.clip(pts_norm, 0.0, 1.0)
        flat = pts_norm.flatten()
        coords_str = " ".join(f"{v:.6f}" for v in flat)
        lines.append(f"{cls} {coords_str}")
    label_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Polygon-aware augmentation
# ---------------------------------------------------------------------------


def _mask_to_polygons(
    mask: NDArr,
    cls: int,
    img_area: float,
) -> list[tuple[int, NDArr]]:
    """Extract simplified polygons from a binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[tuple[int, NDArr]] = []
    min_area = img_area * MIN_AREA_FRAC
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        approx = cv2.approxPolyDP(c, APPROX_EPS_PX, closed=True).reshape(-1, 2)
        if len(approx) < MIN_VERTICES:
            continue
        out.append((cls, approx.astype(np.float32)))
    return out


def _augment(
    img: NDArr,
    polys: list[tuple[int, NDArr]],
    rng: random.Random,
) -> tuple[NDArr, list[tuple[int, NDArr]]]:
    """Apply flip / brightness-contrast / rotation+crop. Polygons round-trip via mask raster."""
    h, w = img.shape[:2]
    img_area = float(h * w)

    # 1) Horizontal flip (probability 0.5).
    if rng.random() < FLIP_PROB:
        img = cv2.flip(img, 1)
        polys = [
            (cls, np.column_stack([w - pts[:, 0], pts[:, 1]]).astype(np.float32))
            for cls, pts in polys
        ]

    # 2) Brightness / contrast (image only -- labels unaffected).
    alpha = rng.uniform(0.8, 1.2)
    beta = rng.randint(-30, 30)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # 3) Rotation +/-10 degrees -- single warpAffine applied to image and to per-instance mask.
    angle = rng.uniform(-10.0, 10.0)
    rot_mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    img_rot = cv2.warpAffine(img, rot_mat, (w, h), borderMode=cv2.BORDER_REFLECT)

    rotated_masks: list[tuple[int, NDArr]] = []
    for cls, pts in polys:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts.astype(np.int32)], color=(1,))
        mask_rot = cv2.warpAffine(  # type: ignore[call-overload]
            mask,
            rot_mat,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        rotated_masks.append((cls, mask_rot))

    # 4) Random scale crop (0.85 - 1.0). Crop both image and every instance mask, then resize.
    scale = rng.uniform(0.85, 1.0)
    if scale < 1.0:
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))
        top = rng.randint(0, h - new_h)
        left = rng.randint(0, w - new_w)
        img_out = cv2.resize(
            img_rot[top : top + new_h, left : left + new_w],
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )
        out_polys: list[tuple[int, NDArr]] = []
        for cls, mask in rotated_masks:
            mask_crop = mask[top : top + new_h, left : left + new_w]
            mask_resized = cv2.resize(mask_crop, (w, h), interpolation=cv2.INTER_NEAREST)
            out_polys.extend(_mask_to_polygons(mask_resized, cls, img_area))
        return img_out, out_polys

    # No crop -- re-extract polygons from rotated masks anyway (rotation may clip near border).
    out_polys = []
    for cls, mask in rotated_masks:
        out_polys.extend(_mask_to_polygons(mask, cls, img_area))
    return img_rot, out_polys


# ---------------------------------------------------------------------------
# Dataset copy + oversample
# ---------------------------------------------------------------------------


def _copy_split(src_root: Path, dst_root: Path, split: str) -> None:
    """Mirror src_root/<split>/{images,labels} into dst_root/<split>/.. (skip existing)."""
    for sub in ("images", "labels"):
        src_dir = src_root / split / sub
        dst_dir = dst_root / split / sub
        if not src_dir.is_dir():
            msg = f"missing source dir: {src_dir}"
            raise FileNotFoundError(msg)
        dst_dir.mkdir(parents=True, exist_ok=True)
        copied = skipped = 0
        for src_path in src_dir.iterdir():
            if not src_path.is_file():
                continue
            dst_path = dst_dir / src_path.name
            if dst_path.exists():
                skipped += 1
                continue
            shutil.copy2(src_path, dst_path)
            copied += 1
        LOGGER.info("[%s/%s] copied=%d skipped=%d", split, sub, copied, skipped)


def _resolve_image_for_label(label_path: Path, img_dir: Path) -> Path | None:
    stem = label_path.stem
    for ext in IMG_EXTS:
        candidate = img_dir / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def _build_class_index(
    img_dir: Path,
    lbl_dir: Path,
) -> tuple[dict[int, list[tuple[Path, Path]]], Counter[int]]:
    """Return {class_id: [(img_path, lbl_path), ...]} and an instance-count Counter."""
    class_to_imgs: dict[int, list[tuple[Path, Path]]] = defaultdict(list)
    counter: Counter[int] = Counter()

    for lbl_path in sorted(lbl_dir.glob("*.txt")):
        # Skip previously-generated oversampled files when *indexing source*; using them as
        # source would compound augmentation noise. They were copied over so they stay
        # available for training, just not as augmentation seeds.
        if "_os_" in lbl_path.stem:
            continue
        img_path = _resolve_image_for_label(lbl_path, img_dir)
        if img_path is None:
            continue
        seen: set[int] = set()
        for line in lbl_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 1 + MIN_VERTICES * 2:
                continue
            try:
                cid = int(parts[0])
            except ValueError:
                continue
            counter[cid] += 1
            if cid not in seen:
                class_to_imgs[cid].append((img_path, lbl_path))
                seen.add(cid)
    return class_to_imgs, counter


def _count_existing_os(img_dir: Path) -> int:
    return sum(1 for p in img_dir.iterdir() if p.is_file() and "_os_" in p.stem)


def _write_data_yaml(src_root: Path, dst_root: Path) -> None:
    src_yaml = src_root / "data.yaml"
    if src_yaml.exists():
        with src_yaml.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}
    cfg.update(
        path=str(dst_root.resolve()),
        train="train/images",
        val="valid/images",
        test="test/images",
        task="segment",
        nc=len(CLASS_NAMES),
        names=CLASS_NAMES,
    )
    dst_yaml = dst_root / "data.yaml"
    with dst_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    LOGGER.info("Wrote %s", dst_yaml)


def _count_all_instances(lbl_dir: Path) -> Counter[int]:
    counter: Counter[int] = Counter()
    for lbl_path in lbl_dir.glob("*.txt"):
        for line in lbl_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 1 + MIN_VERTICES * 2:
                continue
            try:
                counter[int(parts[0])] += 1
            except ValueError:
                continue
    return counter


def _oversample_class(  # noqa: PLR0913 -- helper for one class; splitting further hurts readability
    cid: int,
    needed: int,
    sources: list[tuple[Path, Path]],
    img_dir: Path,
    lbl_dir: Path,
    rng: random.Random,
    os_offset: int,
) -> int:
    generated = 0
    attempt = 0
    max_attempts = max(needed * 20, 200)
    while generated < needed and attempt < max_attempts:
        attempt += 1
        src_img_path, src_lbl_path = rng.choice(sources)
        img = cv2.imread(str(src_img_path))
        if img is None:
            LOGGER.warning("Could not read %s -- skipping.", src_img_path)
            continue
        h, w = img.shape[:2]

        polys = _read_polygons(src_lbl_path, w, h)
        if not polys:
            continue

        aug_img, aug_polys = _augment(img, polys, rng)
        if not any(cls == cid for cls, _ in aug_polys):
            continue

        suffix = f"_os_{os_offset + generated:04d}"
        new_stem = src_img_path.stem + suffix
        new_img_path = img_dir / (new_stem + src_img_path.suffix)
        new_lbl_path = lbl_dir / (new_stem + ".txt")
        if new_img_path.exists():
            continue

        cv2.imwrite(str(new_img_path), aug_img)
        _write_polygons(new_lbl_path, aug_polys, w, h)
        generated += 1
    return generated


def _log_final_counts(
    img_dir: Path,
    lbl_dir: Path,
    instance_counter: Counter[int],
    total_written: int,
) -> None:
    all_counter = _count_all_instances(lbl_dir)
    LOGGER.info("")
    LOGGER.info("Final train polygon instance counts (including _os_ files):")
    LOGGER.info("  %-20s %6s %6s %8s", "class", "before", "after", "delta")
    LOGGER.info("  %s", "-" * 44)
    for cid, name in enumerate(CLASS_NAMES):
        before = instance_counter.get(cid, 0)
        after = all_counter.get(cid, 0)
        LOGGER.info("  %-20s %6d %6d %+8d", name, before, after, after - before)
    n_imgs = sum(1 for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
    LOGGER.info("  total train images: %d", n_imgs)
    LOGGER.info("  total new files written this run: %d", total_written)


def oversample(src_root: Path, dst_root: Path) -> None:
    LOGGER.info("Source dataset : %s", src_root)
    LOGGER.info("Output dataset : %s", dst_root)
    LOGGER.info("Target counts  : %s", TARGET_COUNTS)
    LOGGER.info("")

    if not src_root.is_dir():
        msg = f"source dataset not found: {src_root}"
        raise FileNotFoundError(msg)

    # Mirror dataset structure (copy, not symlink -- Ultralytics resolves real paths when it
    # substitutes /images/ -> /labels/, and a symlink to another dataset silently sends label
    # lookup to the wrong tree).
    LOGGER.info("Copying dataset structure (idempotent)...")
    for split in ("train", "valid", "test"):
        _copy_split(src_root, dst_root, split)
    LOGGER.info("")

    _write_data_yaml(src_root, dst_root)
    LOGGER.info("")

    train_img_dir = dst_root / "train" / "images"
    train_lbl_dir = dst_root / "train" / "labels"
    class_to_imgs, instance_counter = _build_class_index(train_img_dir, train_lbl_dir)

    LOGGER.info("Current polygon instance counts (excluding _os_ files):")
    for cid, name in enumerate(CLASS_NAMES):
        LOGGER.info("  %-20s %4d", name, instance_counter.get(cid, 0))
    LOGGER.info("")

    rng = random.Random(RANDOM_SEED)  # noqa: S311 -- data augmentation, not crypto
    total_written = 0
    os_offset = _count_existing_os(train_img_dir)
    if os_offset:
        LOGGER.info("Found %d existing _os_ files -- new files offset from there.", os_offset)
        LOGGER.info("")

    for class_name, target in TARGET_COUNTS.items():
        if class_name not in CLASS_NAMES:
            LOGGER.warning("Unknown class '%s' -- skipping.", class_name)
            continue
        cid = CLASS_NAMES.index(class_name)
        current = instance_counter.get(cid, 0)
        needed = max(0, target - current)
        if needed == 0:
            LOGGER.info(
                "%-20s already at %d (target %d) -- skipping.", class_name, current, target,
            )
            continue
        sources = class_to_imgs.get(cid, [])
        if not sources:
            LOGGER.warning("No source images contain class '%s' -- cannot oversample.", class_name)
            continue
        LOGGER.info(
            "Oversampling %-20s: %d -> %d (need %d more, from %d unique source images)",
            class_name, current, target, needed, len(sources),
        )
        generated = _oversample_class(
            cid, needed, sources, train_img_dir, train_lbl_dir, rng, os_offset + total_written,
        )
        total_written += generated
        if generated < needed:
            LOGGER.warning("Hit max_attempts for %s -- wrote %d/%d.", class_name, generated, needed)
        else:
            LOGGER.info("  wrote %d new files for %s.", generated, class_name)

    _log_final_counts(train_img_dir, train_lbl_dir, instance_counter, total_written)


def main() -> None:
    oversample(SRC_DATASET.resolve(), DST_DATASET.resolve())
    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
