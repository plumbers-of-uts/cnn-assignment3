---
license: agpl-3.0
library_name: ultralytics
pipeline_tag: image-segmentation
tags:
  - yolo
  - yolo26
  - instance-segmentation
  - object-detection
  - onnx
  - sewage
  - pipe-inspection
  - cctv
base_model: Ultralytics/YOLO26
model-index:
  - name: pipevision-yolo26m-seg
    results:
      - task:
          type: instance-segmentation
        dataset:
          name: Roboflow Sewage Defect Detection v1
          type: image-segmentation
        metrics:
          - type: mAP@0.5 (box)
            value: 0.5336
          - type: mAP@0.5:0.95 (box)
            value: 0.3017
          - type: mAP@0.5 (mask)
            value: 0.4749
          - type: mAP@0.5:0.95 (mask)
            value: 0.2708
---

# pipevision-yolo26m-seg

Unified YOLO26m instance-segmentation model for sewage / plumbing pipe defect
detection. Trained as part of the *Plumbing Defect Detection & Pipe Inspection
System* (UTS CNN Assignment 3, Team **Plumbers of UTS**) and shipped as a
single ONNX file for in-browser inference via the
[pipevision-ai](https://github.com/gracefullight/pipevision-ai) web client.

- **Architecture**: YOLO26m-seg (P3/P4/P5 detection + mask prototypes head)
- **Input**: `float32 [1, 3, 640, 640]`, NCHW, RGB, `0..1`, letterboxed
- **Outputs**: detections `[1, 300, 38]` + prototypes `[1, 32, 160, 160]`
- **Classes** (7): `Buckling`, `Crack`, `Debris`, `Hole`, `Joint offset`, `Obstacle`, `Utility intrusion`
- **Precision**: FP16 (ONNX, opset 17)
- **SHA-256** (`yolo26m-seg-fp16.onnx`):
  `3015a5cca1cce704912aebc01c24d2287af4e07514f279cf81c6cbcc63b4b922`

## Files

| File | Purpose |
|---|---|
| `yolo26m-seg-fp16.onnx` | Web / edge inference (47 MB, FP16, opset 17) |
| `best.pt` | Ultralytics native checkpoint for further training |
| `metadata.yaml` | I/O contract, decode rules, thresholds, metrics |
| `classes.json` | Class ID → name map |
| `per_class_metrics.csv` | Per-class P / R / AP50 / AP for box + mask |
| `results.csv`, `results.png` | Training curves |
| `confusion_matrix*.png`, `test_*.png` | Eval visualisations |
| `qualitative/` | Sample predictions on the test split |

## Test Metrics (Roboflow v1 test split, 98 images / 229 instances)

| Metric | Box | Mask |
|---|---:|---:|
| mAP@0.5 | **0.534** | **0.475** |
| mAP@0.5:0.95 | **0.302** | **0.271** |

### Per-class (test split)

| Class | AP50 (box) | AP (box) | AP50 (mask) | AP (mask) |
|---|---:|---:|---:|---:|
| Utility intrusion | 0.901 | 0.594 | 0.920 | 0.628 |
| Hole | 0.832 | 0.555 | 0.508 | 0.328 |
| Obstacle | 0.704 | 0.347 | 0.683 | 0.376 |
| Debris | 0.597 | 0.260 | 0.604 | 0.313 |
| Crack | 0.397 | 0.208 | 0.266 | 0.109 |
| Joint offset | 0.225 | 0.104 | 0.296 | 0.124 |
| Buckling | 0.080 | 0.044 | 0.048 | 0.016 |

Buckling regression is a known training artefact (`cls_weights` was silently
dropped in Ultralytics 8.4.37); a short fine-tune from `best.pt` with the
8.4.51 trainer and the original Crack 2.6 / Hole 2.4 / Utility 2.8 weights is
the planned fix.

## Mask Decode (TS / Python)

```text
1. detections = output0[:, :, :].T                # [N, 38]
2. boxes, scores, class_ids, coeffs               # split per spec in metadata.yaml
3. Apply confidence threshold (default 0.25) and class-wise NMS (IoU 0.45)
4. proto = output1[0]                              # [32, 160, 160]
5. mask  = sigmoid(coeffs @ proto.reshape(32, 25600)).reshape(160, 160)
6. crop mask to letterboxed bbox, threshold > 0.5, resize to original frame
```

Full client implementation:
[`pipevision-ai/src/lib/postprocess.ts`](https://github.com/gracefullight/pipevision-ai).

## Quick Start — Ultralytics

```python
from ultralytics import YOLO
model = YOLO('best.pt')                            # or yolo26m-seg-fp16.onnx
results = model.predict('pipe_frame.jpg', conf=0.25, iou=0.45)
for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()
    masks = r.masks.data.cpu().numpy()             # [N, H, W]
```

## Quick Start — ONNX Runtime

```python
import numpy as np, onnxruntime as ort
sess = ort.InferenceSession('yolo26m-seg-fp16.onnx')
x = np.random.rand(1, 3, 640, 640).astype(np.float32)
det, proto = sess.run(None, {'images': x})
print(det.shape, proto.shape)   # (1, 300, 38) (1, 32, 160, 160)
```

## Training Recipe

| Knob | Value |
|---|---|
| Base | `yolo26m-seg.pt` (COCO segmentation pretrained) |
| Hardware | SageMaker GPU (ml.g5/g4dn class) |
| Optimizer | MuSGD, cosine LR, AMP on |
| imgsz / batch / epochs | 640 / 8 / 200 (patience 50) |
| Augmentation | mosaic 1.0, close_mosaic 20, mixup 0.10, copy_paste 0.30, scale 0.6, degrees 10, hsv_v 0.30, hsv_s 0.60, fliplr 0.5 |
| Mask labels | SAM 2.1_b pseudo-polygons from YOLO bboxes (see `src/generate_seg_labels.py`) |

## Dataset

- **Source**: [Roboflow Universe — Sewage Defect Detection v1](https://universe.roboflow.com/sewage-defect-detection/sewage-defect-detection)
- **Splits**: 686 train / 196 valid / 98 test
- **Original labels**: bounding boxes; polygon masks generated with SAM 2.1_b
- **Image dimensions**: 640 × 480 (CCTV frames)

## Limitations

- Trained on a small (~1 k images) labelled set; expect drift on cameras with
  very different colour balance or non-CCTV optics (e.g. drone or borescope).
- Buckling AP50 is **0.08** — do not deploy this class without retraining
  with proper class weighting.
- Native input scale is 640 px. SAHI sliced inference is **not recommended**;
  it hurt mAP on the source dataset (see project report §7.5).
- FP16 ONNX: verified box-count parity with FP32 on the test set; expect
  ≤ 1 pp mask mAP delta on borderline thin Cracks.

## Citation

```bibtex
@misc{plumbersofuts2026pipevision,
  title   = {pipevision-yolo26m-seg: instance segmentation for sewage pipe defects},
  author  = {Plumbers of UTS},
  year    = {2026},
  howpublished = {\url{https://huggingface.co/gracefullight/pipevision-yolo26m-seg}}
}
```

## License

AGPL-3.0 (inherited from Ultralytics YOLO26 base weights).
