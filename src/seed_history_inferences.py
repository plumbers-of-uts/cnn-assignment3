"""seed_history_inferences.py — pre-compute model inferences for the web app's
Inspection History seed.

Runs yolo26m-seg `best.pt` on every test image and writes:

    pipevision-ai/public/seed-history/inferences.json   ← list of records
    pipevision-ai/public/seed-history/{slug}.jpg        ← resized thumbnail

The web app's `seed.ts` fetches inferences.json on first launch and inserts
each entry as a `HistoryRecord` so users see ~91 real CCTV inspections with
real predictions instead of synthetic SVG placeholders.

Run from the repo root with:

    uv run python src/seed_history_inferences.py

Idempotent — re-running rebuilds the same files from the same checkpoint.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = REPO_ROOT / "model" / "best.pt"
TEST_IMAGES = REPO_ROOT / "src" / "data" / "sewage-yolo26" / "test" / "images"
WEB_PUBLIC = REPO_ROOT.parent / "pipevision-ai" / "public" / "seed-history"

# Source filenames already used as upload-area samples — excluded so users do
# not see the same frame twice (once in the sample picker, once in History).
SAMPLE_SOURCE_FILES: set[str] = {
    "PI008746_f240_jpg.rf.99618b8c38cb75c23e4db94db28dffb3.jpg",  # buckling
    "PI054350_f15360_jpg.rf.4860de11ea271613e0268399da001b83.jpg",  # crack
    "PI005122_f3300_jpg.rf.37a053102583da9e16fb4b75cbef17a3.jpg",  # debris
    "PI012194_f2040_jpg.rf.848b1eb2bdd50442a2455e85c02e94bd.jpg",  # hole
    "PI009859_f1260_jpg.rf.2d8579877a81e20663bbb6495260b36b.jpg",  # joint-offset
    "PI005237_f540_jpg.rf.42fae19d4612a555229b480d38ea1161.jpg",  # obstacle
    "PI065517_f960_jpg.rf.1d62b2cf1418cb92f63d50dbcc427db6.jpg",  # utility-intrusion
}

# Inference config — match the runtime web client (model-config.ts).
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MAX_DET = 100

# Output assets — JPEG at native 640x480 (no resize) since the test set is
# already that size. Keeps quality close to what the model actually sees.
JPEG_QUALITY = 82


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def slugify(filename: str) -> str:
    """Drop the Roboflow hash suffix and the .jpg extension.

    PI054350_f15360_jpg.rf.4860de11ea271613e0268399da001b83.jpg
        → pi054350_f15360
    """
    stem = filename.split(".rf.")[0]
    if stem.endswith("_jpg"):
        stem = stem[:-4]
    return stem.lower()


def to_record(
    *,
    slug: str,
    filename: str,
    inference_ms: int,
    width: int,
    height: int,
    boxes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Shape matches what `seed.ts` expects."""
    return {
        "slug": slug,
        "source": filename,
        "width": width,
        "height": height,
        "inferenceMs": inference_ms,
        "detections": boxes,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    assert WEIGHTS.exists(), f"model not found: {WEIGHTS}"
    assert TEST_IMAGES.is_dir(), f"test images missing: {TEST_IMAGES}"

    WEB_PUBLIC.mkdir(parents=True, exist_ok=True)
    print(f"[seed] weights: {WEIGHTS}")
    print(f"[seed] images : {TEST_IMAGES}")
    print(f"[seed] output : {WEB_PUBLIC}")

    model = YOLO(str(WEIGHTS))
    print("[seed] model loaded")

    images = sorted(p for p in TEST_IMAGES.glob("*.jpg"))
    images = [p for p in images if p.name not in SAMPLE_SOURCE_FILES]
    print(f"[seed] {len(images)} images to process (sample-7 excluded)")

    records: list[dict[str, Any]] = []

    for i, image_path in enumerate(images, start=1):
        slug = slugify(image_path.name)
        out_jpg = WEB_PUBLIC / f"{slug}.jpg"

        t0 = time.perf_counter()
        result = model.predict(
            source=str(image_path),
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            max_det=MAX_DET,
            verbose=False,
        )[0]
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        # Read image to copy through (and to grab dimensions even if no detections).
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            print(f"[seed] WARN: cv2 could not read {image_path.name} — skipping")
            continue
        h, w = bgr.shape[:2]
        cv2.imwrite(str(out_jpg), bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

        # Boxes come back in xyxy pixel space, classes as int, conf as float.
        det_boxes: list[dict[str, Any]] = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy().astype(int)
            for k in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[k]
                det_boxes.append(
                    {
                        "classId": int(cls[k]),
                        "confidence": round(float(conf[k]), 4),
                        # web app stores bbox as top-left xywh in original-image px
                        "bbox": {
                            "x": int(round(float(x1))),
                            "y": int(round(float(y1))),
                            "w": int(round(float(x2 - x1))),
                            "h": int(round(float(y2 - y1))),
                        },
                    }
                )

        records.append(
            to_record(
                slug=slug,
                filename=image_path.name,
                inference_ms=elapsed_ms,
                width=int(w),
                height=int(h),
                boxes=det_boxes,
            )
        )

        if i % 10 == 0 or i == len(images):
            print(f"[seed] {i}/{len(images)}  last={image_path.name}  dets={len(det_boxes)}  {elapsed_ms} ms")

    manifest_path = WEB_PUBLIC / "inferences.json"
    manifest = {
        "modelVersion": "yolo26m-pipevision-fp16",
        "generatedAt": int(time.time() * 1000),
        "conf": CONF_THRESHOLD,
        "iou": IOU_THRESHOLD,
        "records": records,
    }
    manifest_path.write_text(json.dumps(manifest, separators=(",", ":")))
    total_dets = sum(len(r["detections"]) for r in records)
    print(
        f"[seed] wrote {manifest_path} — {len(records)} records, "
        f"{total_dets} total detections, {sum(r['inferenceMs'] for r in records)} ms cumulative"
    )


if __name__ == "__main__":
    main()
