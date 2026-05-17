# Plan — Unified yolo26m-seg Model (cnn-assignment3 scope)

**Status**: design approved, ready for `/plan`
**Owner**: cnn-assignment3 repo
**Pair plan**: `pipevision-ai/docs/plans/yolo26m-seg-web-integration.md`

This plan covers steps 1–5 of the unified-segmentation initiative:

1. SAM pseudo-label generation
2. Training data assembly (`sewage-yolo26-seg/`)
3. SageMaker GPU training (yolo26m-seg)
4. ONNX FP16 export (dual output)
5. HuggingFace Hub upload

Steps 6–12 (web app modifications) live in the pipevision-ai plan.

---

## Interface Contract — shared with pipevision-ai

Both plans MUST agree on this contract; either side can change it only via paired update.

| Item | Value |
|---|---|
| HF model repo | `gracefullight/pipevision-yolo26m-seg` |
| ONNX file name | `yolo26m-seg-fp16.onnx` |
| Input | `images: float32 [1, 3, 640, 640]`, NCHW, RGB, `0..1`, letterboxed |
| Output 0 — detections | `float32 [1, 300, 38]` (post-NMS, fixed max_det=300) |
|   cols 0..3 | x, y, w, h in original pixel space (letterboxed 640 grid) |
|   col 4 | confidence (already sigmoided × class prob) |
|   col 5 | class_id (float, cast to int) |
|   cols 6..37 | 32 mask coefficients (no activation) |
| Output 1 — prototypes | `float32 [1, 32, 160, 160]` |
| Mask decode | `mask = sigmoid(coefficients @ prototypes.reshape(32, 25600)).reshape(160, 160)` → threshold 0.5 → crop to bbox → resize to original |
| Class IDs | 0 Buckling, 1 Crack, 2 Debris, 3 Hole, 4 Joint offset, 5 Obstacle, 6 Utility intrusion |
| Inference defaults | conf=0.25, iou=0.45 (NMS embedded) |
| NMS | **embedded** in ONNX graph — TS client only filters by conf and unpads zero-rows |
| Opset | 17 |
| Precision | FP16 weights, FP32 I/O |
| SHA-256 | `3015a5cca1cce704912aebc01c24d2287af4e07514f279cf81c6cbcc63b4b922` |

If any contract field changes, **bump the HF file name** (`-v2`) and update both plans.

---

## Phase 1 — SAM Pseudo-Label Generator

**New script**: `src/generate_seg_labels.py`

Inputs: `src/data/sewage-yolo26/{train,valid,test}/{images,labels}/`
Outputs: `src/data/sewage-yolo26-seg/{train,valid,test}/{images,labels}/`

For each image:
1. Read YOLO bbox labels.
2. Call `ultralytics.SAM('sam2.1_b.pt')` with the bboxes as prompts.
3. For each returned binary mask:
   - `cv2.findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)`
   - Keep the largest contour by area.
   - `cv2.approxPolyDP(contour, epsilon=2.0)` to reduce point count.
   - Skip if polygon has < 3 points after simplification.
4. Normalise polygon points by image width/height.
5. Write `cls x1 y1 x2 y2 ... xn yn` per defect, one line per detection.
6. Symlink original image into output dir (no copy).

Edge cases:
- Empty label file → skip (no seg file written).
- SAM returns empty mask → skip that detection.
- Multiple disconnected components → keep only the largest.

Validation script `tests/test_generate_seg_labels.py`:
- 5-image sample, assert polygon counts match bbox counts.
- Compare bbox derived from polygon to original bbox (IoU ≥ 0.8).

**Time**: ~30 min coding + ~30 min full-dataset run on local MPS (or ~5 min on SageMaker T4).

## Phase 2 — Dataset Assembly

**New file**: `src/data/sewage-yolo26-seg/data.yaml`

```yaml
path: ../sewage-yolo26-seg
train: train/images
val:   valid/images
test:  test/images
nc: 7
names: ['Buckling', 'Crack', 'Debris', 'Hole', 'Joint offset', 'Obstacle', 'Utility intrusion']
task: segment
```

Smoke check: `uv run python -c "from ultralytics.data.utils import check_det_dataset; print(check_det_dataset('src/data/sewage-yolo26-seg/data.yaml'))"`.

Make sure `.gitignore` excludes the new seg labels (same convention as `sewage-yolo26/`).

## Phase 3 — SageMaker Seg Training

**New notebook**: `src/sagemaker_seg_train.ipynb` (clone existing `sagemaker_train.ipynb`).

Key changes vs detection notebook:

```python
model = YOLO('yolo26-seg.yaml')
try:
    model.load('yolo26m-seg.pt')   # COCO seg pretrained
except Exception:
    pass

train_kwargs = dict(
    data=data_yaml,               # seg data.yaml
    task='segment',
    epochs=200, imgsz=640, batch=8 if vram_gb < 20 else 16,
    optimizer='MuSGD', cos_lr=True, patience=50,
    mosaic=1.0, close_mosaic=20, mixup=0.10, copy_paste=0.30,
    scale=0.6, degrees=10.0, translate=0.10,
    hsv_v=0.30, hsv_s=0.60, fliplr=0.5,
    cls_weights=cls_weights,      # 8.4.51 accepts
    amp=True, workers=4,
    project='runs/segment', name='tier2_seg',
)
results = model.train(**train_kwargs)
```

Validation cell prints **both** box mAP and mask mAP per class. Stage to S3 via the same shutdown cell.

**Time**: ~2-3 h on ml.g4dn.xlarge T4 ($1.85), ~1.5 h on g5.xlarge A10G ($1.80).

## Phase 4 — ONNX Export

**Update `pipevision-ai/model/export.py`** (or copy script into this repo and adapt) — the export call must produce the dual-output ONNX:

```python
exported_path = model.export(
    format='onnx',
    half=True,
    dynamic=False,
    simplify=True,
    imgsz=640,
    opset=17,
    nms=True,                  # end-to-end NMS embedded, max_det=300
)
```

Ultralytics auto-detects `task=segment` from the .pt and emits the dual output. Verify with `onnx.checker.check_model(loaded)` + visualise on netron.app — confirm 2 outputs, second of shape `[1, 32, 160, 160]` and first of shape `[1, 300, 38]` (4 bbox + conf + class_id + 32 mask coeffs).

Smoke-test in Python:

```python
import onnxruntime as ort
import numpy as np
sess = ort.InferenceSession('yolo26m-seg-fp16.onnx')
out = sess.run(None, {'images': np.random.rand(1, 3, 640, 640).astype(np.float32)})
print([o.shape for o in out])   # expect [(1, 300, 38), (1, 32, 160, 160)]
```

## Phase 5 — HuggingFace Hub Upload

Use `huggingface_hub` SDK (already a transitive dep via ultralytics). Token via `HF_TOKEN` env var.

```python
from huggingface_hub import HfApi, create_repo

api = HfApi()
repo_id = 'gracefullight/pipevision-yolo26m-seg'
create_repo(repo_id, private=False, exist_ok=True)

api.upload_folder(
    folder_path='runs/segment/tier2_seg/weights',
    repo_id=repo_id,
    repo_type='model',
    commit_message='unified seg model',
)
```

Repo contents:
- `yolo26m-seg-fp16.onnx`
- `best.pt` (backup)
- `README.md` (auto-generated model card — see template below)
- `classes.json`
- `metadata.yaml`

Compute and publish SHA-256:

```bash
sha256sum yolo26m-seg-fp16.onnx | awk '{print $1}'
```

Update `pipevision-ai/.env.example` and pair plan's contract section with this hash. **This is the handoff point** to the web team.

Model card template (`README.md`):

```markdown
# pipevision-yolo26m-seg

Unified YOLO26m instance-segmentation model for sewage pipe defect detection.

## Classes
0 Buckling, 1 Crack, 2 Debris, 3 Hole, 4 Joint offset, 5 Obstacle, 6 Utility intrusion

## Training
- Base: yolo26m-seg.pt (COCO segmentation pretrained)
- Dataset: Roboflow Sewage Defect Detection (980 imgs, bbox labels + SAM2.1_b pseudo polygons)
- Optimizer: MuSGD, cosine LR, 200 epochs, patience 50
- Aug: mosaic 1.0, copy_paste 0.3, mixup 0.1, cls_weights

## Metrics (test split)
| Metric | Value |
| ... | ... |     <!-- fill after training -->

## Inputs / Outputs
See `metadata.yaml`.
```

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| SAM masks bleed on thin Cracks → bad polygons | Manually inspect 20 random train samples after Phase 1; tighten `epsilon` if needed |
| yolo26m-seg.pt download fails on SageMaker | Pre-stage to S3 during dataset upload |
| FP16 export quality drop on small Cracks | Compare FP16 vs FP32 mask mAP in Phase 4 smoke test; fall back to FP32 if delta > 5pp |
| HF upload bandwidth | Use S3 → HF mirror if direct upload from SageMaker is slow |
| Contract drift | Both plans cite the same hash; CI check on web side warns when local ONNX hash doesn't match VITE_MODEL_SHA256 |

## Quality gate (per `.claude/rules/python-quality-gate.md`)

After Phase 1 script lands: `uv run poe lint && uv run poe type-check`.

## Definition of done

- [x] `src/generate_seg_labels.py` runs on full dataset, produces polygon labels for ≥ 95% of bbox annotations.
- [x] `runs/segment/tier2_seg/weights/best.pt` exists with test mask mAP@0.5 ≥ 0.40 (achieved **0.475**).
- [x] `yolo26m-seg-fp16.onnx` passes the smoke-test (correct output shapes `[1, 300, 38]` + `[1, 32, 160, 160]`).
- [x] HuggingFace repo public ([`gracefullight/pipevision-yolo26m-seg`](https://huggingface.co/gracefullight/pipevision-yolo26m-seg)), with model card and `metadata.yaml`.
- [x] SHA-256 hash (`3015a5cc…`) committed to **both** plans.
- [ ] `docs/cnn-ass3-PartC-experimental-report.md` extended with seg results section.
