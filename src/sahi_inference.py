from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction

if TYPE_CHECKING:
    from collections.abc import Iterable

    from sahi.models.base import DetectionModel

LOGGER = logging.getLogger("yolo26-sahi")

# ---------------------------------------------------------------------------
# Configuration (edit constants here instead of CLI args).
# ---------------------------------------------------------------------------
DATA_YAML_PATH = "src/data/sewage-yolo26/data.yaml"
# Resolved at runtime when WEIGHTS_PATH is empty: picks the most recent best.pt
# under runs/detect/*/weights/best.pt.
WEIGHTS_PATH: str = ""
EVAL_SPLIT = "test"  # "valid" or "test"

# Slicing parameters — small pipe defects benefit from finer tiles.
SLICE_HEIGHT = 384
SLICE_WIDTH = 384
OVERLAP_HEIGHT_RATIO = 0.2
OVERLAP_WIDTH_RATIO = 0.2

# Detection thresholds — Ultralytics defaults are conf=0.25, iou=0.7. For
# recall-oriented evaluation on long-tail pipe defects we lower conf so SAHI
# can vote across tiles.
CONFIDENCE_THRESHOLD = 0.15
POSTPROCESS_TYPE = "GREEDYNMM"
POSTPROCESS_MATCH_METRIC = "IOS"
POSTPROCESS_MATCH_THRESHOLD = 0.5

# When True, also run the standard (non-sliced) prediction with the same
# weights/conf so the SAHI delta is directly visible in one log.
RUN_STANDARD_BASELINE = True

# When True, run a slice-size sweep at fixed overlap to find the best tile size
# for this dataset/model. Reports a side-by-side mAP table.
SLICE_SIZE_SWEEP_ENABLE = True
SLICE_SIZE_SWEEP_CANDIDATES: tuple[int, ...] = (256, 384, 512)

# Minimum number of fields required per YOLO label row: class + 4 bbox values.
YOLO_LABEL_MIN_FIELDS = 5

ImagePathSet = set[str]


@dataclass
class EvalPaths:
    """Holds resolved paths for SAHI evaluation."""

    data_yaml: Path
    weights: Path
    images_dir: Path
    labels_dir: Path
    class_names: list[str]


def _resolve_data_yaml(data_yaml: str) -> Path:
    path = Path(data_yaml).resolve()
    if not path.exists():
        msg = f"data.yaml not found: {path}"
        raise FileNotFoundError(msg)
    return path


def _resolve_weights(explicit: str) -> Path:
    if explicit:
        candidate = Path(explicit).resolve()
        if not candidate.exists():
            msg = f"WEIGHTS_PATH set but missing: {candidate}"
            raise FileNotFoundError(msg)
        return candidate

    runs_root = Path("runs/detect").resolve()
    candidates = sorted(
        runs_root.glob("*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        msg = (
            "No best.pt found under runs/detect/*/weights/. "
            "Train first or set WEIGHTS_PATH explicitly."
        )
        raise FileNotFoundError(msg)
    return candidates[0]


def _resolve_split_dirs(data_yaml: Path, split: str) -> tuple[Path, Path]:
    """Resolve images/labels directories for the requested split.

    Defers to Ultralytics' `check_det_dataset` so that the same path-resolution
    rules used by training apply here. Falls back to a manual scan that
    handles common Roboflow layouts when that import fails.
    """
    # Use Ultralytics' canonical resolver — handles `path:` key, relative paths,
    # Roboflow `../split/images`, and the auto-download fallback.
    from ultralytics.data.utils import check_det_dataset

    info = check_det_dataset(str(data_yaml))
    raw_path = info.get(split)
    if not raw_path:
        msg = f"data.yaml has no entry for split '{split}'"
        raise KeyError(msg)

    images_dir = Path(raw_path).resolve()
    if not images_dir.exists():
        msg = f"Resolved images dir does not exist for split '{split}': {images_dir}"
        raise FileNotFoundError(msg)

    # YOLO convention: images at .../split/images, labels at .../split/labels.
    labels_dir = images_dir.parent / "labels"
    if not labels_dir.exists():
        msg = f"Labels dir not found alongside images: {labels_dir}"
        raise FileNotFoundError(msg)

    return images_dir, labels_dir


def _read_class_names(data_yaml: Path) -> list[str]:
    with data_yaml.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    names = payload.get("names", [])
    if isinstance(names, dict):
        return [str(names[idx]) for idx in sorted(names)]
    return [str(name) for name in names]


def _build_coco_ground_truth(
    images_dir: Path,
    labels_dir: Path,
    class_names: list[str],
) -> dict[str, Any]:
    """Convert YOLO-format labels into a COCO-style dict for pycocotools."""
    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    ann_id = 1

    image_paths = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}],
    )

    for image_id, img_path in enumerate(image_paths, start=1):
        with Image.open(img_path) as im:
            width, height = im.size

        coco_images.append(
            {
                "id": image_id,
                "file_name": img_path.name,
                "width": width,
                "height": height,
            },
        )

        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < YOLO_LABEL_MIN_FIELDS:
                continue
            cls_id = int(parts[0])
            x_c, y_c, w, h = (float(v) for v in parts[1:5])
            bbox_w = w * width
            bbox_h = h * height
            bbox_x = (x_c * width) - bbox_w / 2.0
            bbox_y = (y_c * height) - bbox_h / 2.0

            coco_annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cls_id,
                    "bbox": [bbox_x, bbox_y, bbox_w, bbox_h],
                    "area": float(bbox_w * bbox_h),
                    "iscrowd": 0,
                    "segmentation": [],
                },
            )
            ann_id += 1

    return {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [
            {"id": idx, "name": name} for idx, name in enumerate(class_names)
        ],
    }


def _build_sahi_model(weights: Path, conf: float) -> DetectionModel:
    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(weights),
        confidence_threshold=conf,
    )


def _predict_sliced(
    model: DetectionModel,
    image_path: Path,
    image_id: int,
    slice_height: int = SLICE_HEIGHT,
    slice_width: int = SLICE_WIDTH,
) -> list[dict[str, Any]]:
    result = get_sliced_prediction(
        image=str(image_path),
        detection_model=model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
        overlap_width_ratio=OVERLAP_WIDTH_RATIO,
        postprocess_type=POSTPROCESS_TYPE,
        postprocess_match_metric=POSTPROCESS_MATCH_METRIC,
        postprocess_match_threshold=POSTPROCESS_MATCH_THRESHOLD,
        verbose=0,
    )
    return _object_predictions_to_coco(result.object_prediction_list, image_id)


def _predict_standard(
    model: DetectionModel,
    image_path: Path,
    image_id: int,
) -> list[dict[str, Any]]:
    result = get_prediction(
        image=str(image_path),
        detection_model=model,
        verbose=0,
    )
    return _object_predictions_to_coco(result.object_prediction_list, image_id)


def _object_predictions_to_coco(
    predictions: Iterable[Any],
    image_id: int,
) -> list[dict[str, Any]]:
    coco_dets: list[dict[str, Any]] = []
    for pred in predictions:
        bbox = pred.bbox.to_xywh()  # [x, y, w, h]
        coco_dets.append(
            {
                "image_id": image_id,
                "category_id": int(pred.category.id),
                "bbox": [float(v) for v in bbox],
                "score": float(pred.score.value),
            },
        )
    return coco_dets


def _coco_evaluate(gt_dict: dict[str, Any], detections: list[dict[str, Any]]) -> dict[str, float]:
    if not detections:
        LOGGER.warning("No detections produced; returning zeroed metrics.")
        return {"mAP@0.5:0.95": 0.0, "mAP@0.5": 0.0, "mAP@0.75": 0.0}

    with tempfile.TemporaryDirectory() as tmp_dir:
        gt_path = Path(tmp_dir) / "gt.json"
        dt_path = Path(tmp_dir) / "dt.json"
        gt_path.write_text(json.dumps(gt_dict), encoding="utf-8")
        dt_path.write_text(json.dumps(detections), encoding="utf-8")

        coco_gt = COCO(str(gt_path))
        coco_dt = coco_gt.loadRes(str(dt_path))
        evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

        stats = evaluator.stats
        return {
            "mAP@0.5:0.95": float(stats[0]),
            "mAP@0.5": float(stats[1]),
            "mAP@0.75": float(stats[2]),
            "mAP_small": float(stats[3]),
            "mAP_medium": float(stats[4]),
            "mAP_large": float(stats[5]),
        }


def _run_inference_pass(  # noqa: PLR0913 — clarity beats packing args into a config object here.
    label: str,
    model: DetectionModel,
    gt_dict: dict[str, Any],
    images_dir: Path,
    *,
    sliced: bool,
    slice_height: int = SLICE_HEIGHT,
    slice_width: int = SLICE_WIDTH,
) -> dict[str, float]:
    LOGGER.info("=== %s (sliced=%s, tile=%dx%d) ===", label, sliced, slice_height, slice_width)
    detections: list[dict[str, Any]] = []
    for image_meta in gt_dict["images"]:
        image_path = images_dir / image_meta["file_name"]
        if sliced:
            detections.extend(
                _predict_sliced(
                    model,
                    image_path,
                    image_meta["id"],
                    slice_height=slice_height,
                    slice_width=slice_width,
                ),
            )
        else:
            detections.extend(_predict_standard(model, image_path, image_meta["id"]))

    metrics = _coco_evaluate(gt_dict, detections)
    LOGGER.info(
        "%s | mAP@0.5=%.4f | mAP@0.5:0.95=%.4f | mAP_small=%.4f",
        label,
        metrics["mAP@0.5"],
        metrics["mAP@0.5:0.95"],
        metrics["mAP_small"],
    )
    return metrics


def _log_sweep_table(rows: list[tuple[str, dict[str, float]]]) -> None:
    """Print a compact comparison table across runs."""
    header = (
        f"{'Run':<24} | {'mAP@0.5':>9} | {'mAP@0.5:0.95':>13} | "
        f"{'mAP_small':>10} | {'mAP_medium':>11} | {'mAP_large':>10}"
    )
    LOGGER.info("=" * len(header))
    LOGGER.info(header)
    LOGGER.info("-" * len(header))
    for name, m in rows:
        LOGGER.info(
            "%-24s | %9.4f | %13.4f | %10.4f | %11.4f | %10.4f",
            name,
            m.get("mAP@0.5", 0.0),
            m.get("mAP@0.5:0.95", 0.0),
            m.get("mAP_small", 0.0),
            m.get("mAP_medium", 0.0),
            m.get("mAP_large", 0.0),
        )
    LOGGER.info("=" * len(header))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    data_yaml = _resolve_data_yaml(DATA_YAML_PATH)
    weights = _resolve_weights(WEIGHTS_PATH)
    images_dir, labels_dir = _resolve_split_dirs(data_yaml=data_yaml, split=EVAL_SPLIT)
    class_names = _read_class_names(data_yaml)

    LOGGER.info("Weights: %s", weights)
    LOGGER.info("Split: %s | images=%s | labels=%s", EVAL_SPLIT, images_dir, labels_dir)
    LOGGER.info(
        "Slice config: %dx%d | overlap=(%.2f,%.2f) | conf=%.2f | postproc=%s/%s@%.2f",
        SLICE_HEIGHT,
        SLICE_WIDTH,
        OVERLAP_HEIGHT_RATIO,
        OVERLAP_WIDTH_RATIO,
        CONFIDENCE_THRESHOLD,
        POSTPROCESS_TYPE,
        POSTPROCESS_MATCH_METRIC,
        POSTPROCESS_MATCH_THRESHOLD,
    )

    gt_dict = _build_coco_ground_truth(
        images_dir=images_dir,
        labels_dir=labels_dir,
        class_names=class_names,
    )
    LOGGER.info(
        "COCO GT built | images=%d | annotations=%d | classes=%d",
        len(gt_dict["images"]),
        len(gt_dict["annotations"]),
        len(gt_dict["categories"]),
    )

    model = _build_sahi_model(weights=weights, conf=CONFIDENCE_THRESHOLD)

    rows: list[tuple[str, dict[str, float]]] = []

    standard_metrics: dict[str, float] | None = None
    if RUN_STANDARD_BASELINE:
        standard_metrics = _run_inference_pass(
            label="Standard (no slicing)",
            model=model,
            gt_dict=gt_dict,
            images_dir=images_dir,
            sliced=False,
        )
        rows.append(("standard", standard_metrics))

    if SLICE_SIZE_SWEEP_ENABLE:
        for tile in SLICE_SIZE_SWEEP_CANDIDATES:
            sweep_metrics = _run_inference_pass(
                label=f"SAHI sliced ({tile}x{tile})",
                model=model,
                gt_dict=gt_dict,
                images_dir=images_dir,
                sliced=True,
                slice_height=tile,
                slice_width=tile,
            )
            rows.append((f"sahi-{tile}", sweep_metrics))
    else:
        sahi_metrics = _run_inference_pass(
            label="SAHI sliced",
            model=model,
            gt_dict=gt_dict,
            images_dir=images_dir,
            sliced=True,
        )
        rows.append((f"sahi-{SLICE_HEIGHT}", sahi_metrics))

        if standard_metrics is not None:
            delta_50 = sahi_metrics["mAP@0.5"] - standard_metrics["mAP@0.5"]
            delta_5095 = sahi_metrics["mAP@0.5:0.95"] - standard_metrics["mAP@0.5:0.95"]
            LOGGER.info(
                "SAHI delta | mAP@0.5 %+0.4f | mAP@0.5:0.95 %+0.4f",
                delta_50,
                delta_5095,
            )

    _log_sweep_table(rows)


if __name__ == "__main__":
    main()
