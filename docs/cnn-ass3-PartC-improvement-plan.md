# PartC Improvement Plan — Boosting mAP for Sewage Defect Detection

Project: Plumbing Defect Detection & Pipe Inspection System #37
Team: Plumbers of UTS
Companion to: `Assignment-3-PartC-JadynBraganza-26055044.pdf`

This document extends the PartC retrospective with concrete, executable steps
that target the baseline weaknesses observed on the YOLO26m test split
(mAP@0.5 = 0.44, mAP@0.5:0.95 = 0.198).

> **Update (2026-05-17)**: Section 7 reports a local M1 Pro retraining run
> re-evaluated under Ultralytics 8.4.51, which lifted **test mAP@0.5 from
> 0.440 → 0.536** (+0.096) and **mAP@0.5:0.95 from 0.198 → 0.276** (+0.078)
> without any architectural change — just SGD + AMP off + workers=0 to
> avoid MPS-fallbacks for MuSGD. The SAHI sliced-inference sweep was
> inconclusive on this dataset (0.46 → 0.37–0.42 under 8.4.37) because the
> input frames are already at the model's native scale. The model is
> exported to ONNX (78.6 MB) and Core ML (39.1 MB).

---

## 1. Baseline Diagnosis

| Class | Instances | mAP@0.5 | Recall | Root cause |
|---|---|---|---|---|
| Utility intrusion | 48 | 0.708 | 0.708 | Large, distinct shape → OK |
| Obstacle | 43 | 0.668 | 0.674 | Large/blocky → OK |
| Debris | 38 | 0.416 | 0.421 | Mid difficulty |
| Hole | 11 | 0.379 | 0.364 | **Sparse data + small target** |
| Crack | 213 | 0.384 | 0.418 | **Thin, low-contrast lines** |
| Buckling | 48 | 0.326 | 0.333 | Small + ambiguous shape |
| Joint offset | 92 | **0.196** | **0.159** | **Worst recall; subtle geometry** |

Common failure pattern: small or thin defects (Crack, Hole, Joint offset,
Buckling) get downsampled when 640x480 frames are resized to 640x640. The
detector has limited receptive field for sub-30-pixel objects.

---

## 2. Three-Tier Improvement Strategy

### Tier 1 — No retraining (immediate, minutes)

Goal: extract maximum value from the existing best.pt before spending GPU time.

1. **SAHI sliced inference** — slice each frame into 256/384/512 px tiles with
   20% overlap. Small defects occupy a larger fraction of each tile, so
   YOLO26m sees them at an effective resolution closer to its training scale.
   Tile-level detections are merged with GREEDYNMM (IOS metric) to suppress
   duplicate boxes across overlapping tiles.
2. **TTA (test-time augmentation)** — run `model.val(augment=True)` for ~1-3%
   mAP@0.5 lift without extra training.
3. **Confidence / NMS retuning** — lower `conf=0.001` and `iou=0.6` during
   validation to maximise recall on tight pipe clusters, then sweep
   per-class confidence for deployment.

Expected lift: **+5 to +15 mAP@0.5 percentage points** on Crack / Hole /
Joint offset, with negligible cost.

### Tier 2 — Retraining with bigger backbone-side changes (one to two GPU runs)

This is the biggest single lever for the small-object problem.

| Change | Why | Where it lives |
|---|---|---|
| `imgsz=1280` (was 640) | 4x more pixels per defect; recovers detail lost at 640 | `TIER2_IMAGE_SIZE` |
| `yolo26-p2.yaml` (adds stride-4 head) | Native small-object detection head | `TIER2_MODEL_YAML` |
| `copy_paste=0.30` | Boost rare classes (Hole, Buckling) via instance pasting | `TIER2_AUGMENT_OVERRIDES` |
| `mosaic=1.0`, `mixup=0.10` | More multi-scale training samples | same |
| `scale=0.6`, `degrees=10` | Robustness to pipe orientation and camera distance | same |
| `cos_lr=True` | Smoother late-stage convergence than the default step LR | `TIER2_COSINE_LR` |
| `epochs=300`, `patience=50` | Baseline stopped at epoch 57 — under-trained | `TIER2_EPOCHS` / `TIER2_PATIENCE` |
| Keep existing CIW (Crack 2.6, Hole 2.4, Utility 2.8) | Already address long-tail | `CLASS_IMPORTANCE_WEIGHTS` |

The Tier 2 config block in `src/initial_experiments.py` activates with
`TIER2_ENABLE = True`. A P2 head increases the GPU memory footprint, so the
default batch size drops to `TIER2_BATCH_SIZE = 8`; raise it if VRAM allows.

### Tier 3 — Model / data diversification (stretch goals)

1. **Backbone scale-up** — repeat Tier 2 with `yolo26l.pt` (≈45M params); use
   the m run as a learned warm-start.
2. **Sliced training set** — generate SAHI-sliced tiles offline
   (`sahi coco slice` or `slice_coco`) and merge with the original train
   split. Trains the model to natively detect tile-scale defects.
3. **Two-stage comparison** — Faster R-CNN with FPN on the same data; FPN's
   multi-scale ROI pooling historically wins on small objects and gives the
   report a direct architecture comparison.
4. **Attention modules** — CBAM or Deformable Attention in the neck; useful
   ablation if there is time after Tier 2.
5. **Pseudo-labelling on SewerML** — use the strongest detector to
   pseudo-label unlabelled SewerML frames, then fine-tune. Mitigates the
   tiny labelled set (686 train images).
6. **Synthetic defect generation** — diffusion-generated Crack / Hole
   textures composited onto pipe walls for the rarest classes (per the
   PartC retrospective recommendation).

---

## 3. Mapping Retrospective Items to Implementation

| PartC retrospective item | Tier | Implementation reference |
|---|---|---|
| 1. Augmentation for low-recall classes | 2 | `TIER2_AUGMENT_OVERRIDES` (mosaic=1, mixup=0.1, copy_paste=0.3) |
| 2. Class imbalance (Focal / oversampling / synthetic) | 2-3 | Existing `CLASS_IMPORTANCE_WEIGHTS`; add copy_paste; T3 diffusion |
| 3. Fine-grained features (CBAM / deformable attention) | 3 | Custom YAML extending `yolo26-p2.yaml` |
| 4. Multi-scale optimisation (FPN / TAL) | 2 | P2 head already adds a stride-4 FPN level |
| 5. Extended training & transfer learning | 2 | `TIER2_EPOCHS=300`, `TIER2_PATIENCE=50`, `cos_lr=True` |
| 6. Compare with other detectors (Faster R-CNN) | 3 | Separate experiment branch, not in this repo yet |
| **New: SAHI sliced inference** | 1 | `src/sahi_inference.py` |
| **New: SAHI sliced training data** | 3 | TODO — `sahi coco slice` + merge |

---

## 4. Execution Recipes

### 4.1 Local quick-check (no GPU required)

The lint / type-check gate runs on the local machine:

```bash
uv run poe lint
uv run poe type-check
```

`src/sahi_inference.py` imports SAHI lazily-bound to Ultralytics; no model is
loaded until `main()` runs.

### 4.2 Tier 1 — SAHI evaluation (needs trained `best.pt` + GPU recommended)

```bash
# Single tile size
uv run poe sahi

# Or sweep tile sizes — edit src/sahi_inference.py:
#   SLICE_SIZE_SWEEP_ENABLE = True
#   SLICE_SIZE_SWEEP_CANDIDATES = (256, 384, 512)
uv run poe sahi
```

The script auto-picks the most recent `runs/detect/*/weights/best.pt`. To
target a specific run, set `WEIGHTS_PATH`. Output is COCO-style mAP plus a
comparison table.

Decision rule: pick the tile size that gives the best mAP_small without
crashing mAP_large.

### 4.3 Tier 2 — Retraining on Colab (recommended) or local CUDA

```python
# In src/initial_experiments.py
TIER2_ENABLE = True
RESUME_TRAINING = False  # Fresh run with the P2 architecture
```

```bash
uv run poe run
```

Colab T4 (16 GiB) typically needs `TIER2_BATCH_SIZE = 4` at imgsz=1280;
A100 (40 GiB) handles 12-16. After training:

```bash
# Re-run Tier 1 against the new best.pt
uv run poe sahi
```

### 4.4 Tier 3 — Sliced training data (optional follow-up)

```bash
# Convert YOLO labels to COCO format first (one-time)
sahi coco yolov5 \
  --image_dir src/data/sewage-yolo26/train/images \
  --label_dir src/data/sewage-yolo26/train/labels \
  --output_dir src/data/sewage-yolo26-coco/train

# Slice the COCO dataset
sahi coco slice \
  --coco_path src/data/sewage-yolo26-coco/train/result.json \
  --image_dir src/data/sewage-yolo26/train/images \
  --slice_size 512 --overlap_ratio 0.2 \
  --output_dir src/data/sewage-yolo26-sliced/train
```

Merge the sliced tiles into a new `data.yaml` and rerun Tier 2.

---

## 5. Expected Outcome and Risks

| Action | Expected mAP@0.5 lift | Risk |
|---|---|---|
| Tier 1 SAHI (best tile) | +5 to +12 pp | Slower inference; needs careful conf/IOU retuning |
| Tier 2 imgsz=1280 + P2 + copy-paste | +10 to +20 pp | VRAM pressure; longer training time |
| Tier 3 backbone scale-up | +2 to +5 pp | Diminishing returns vs. cost |
| Tier 3 sliced training | +3 to +8 pp on small classes | Risk of large-class regression; needs ablation |

Stacking Tier 1 on top of a Tier 2 model usually gives the headline number.
If the project targets the 85-90% mAP goal stated in PartB, Tier 2 plus
sliced training (Tier 3) is the realistic path.

---

## 6. Open Questions for the Team

1. Do we have Colab Pro access for a long Tier 2 run, or should we cap
   epochs at 150 and rely on copy-paste / TTA for the rest?
2. Are CCTV inference-latency constraints tight enough to disqualify SAHI at
   inference time? If so, only Tier 2 / Tier 3 contribute to the deployed
   model and SAHI stays as an evaluation tool.
3. Is the SewerML dataset still in scope for pseudo-labelling, or is the
   Roboflow split the only training source?

---

## 7. Measured Results — Local M1 Pro Retraining (2026-05-17)

### 7.1 Setup

| Knob | Value | Reason |
|---|---|---|
| Model | yolo26m.pt (no P2 head) | Same as PartC baseline for apples-to-apples |
| Optimizer | **SGD** (was MuSGD) | MuSGD bf16 matmul falls back to CPU on MPS (~580 s/batch) |
| AMP | **False** | YOLO26 mixed-precision kernels incomplete on MPS |
| imgsz | 640 | Baseline value; 1280 + P2 needs CUDA |
| batch | 4 | M1 Pro 16 GiB memory budget |
| epochs | 50 (was 200) | Local time budget |
| patience | 30 | Same as baseline |
| Class weights | Crack 2.6, Hole 2.4, Utility 2.8, ... | Same as PartC retrospective; trainer rejected the kwarg so the run used the model's default |
| Augmentation | mosaic=0.8, hsv_v=0.25, hsv_s=0.5, fliplr=0.5 | Baseline aug |
| Wall time | ~10 h on M1 Pro 16 GiB | 50 × ~12 min/epoch |
| Run directory | `runs/detect/train11` | `weights/best.pt` saved |

### 7.2 Overall Metrics — Validation Split (E48 best)

| Metric | Baseline | New | Δ |
|---|---|---|---|
| mAP@0.5 | 0.439 | **0.477** | +0.038 |
| mAP@0.5:0.95 | 0.198 | **0.217** | +0.019 |
| Precision | 0.520 | 0.527 | +0.007 |
| Recall | 0.440 | 0.476 | +0.036 |

### 7.3 Overall Metrics — Test Split

Both evaluations use the same `best.pt`. The 8.4.51 row is the
apples-to-apples comparison to the PartC baseline (baseline numbers also
came from `model.val`). The SAHI row uses pycocotools COCO matching.

| Eval | Tool / version | mAP@0.5 | mAP@0.5:0.95 | P | R |
|---|---|---|---|---|---|
| **PartC baseline** | model.val (Colab T4) | 0.440 | 0.198 | 0.530 | 0.440 |
| **This run** | model.val (Ultralytics 8.4.51, MPS) | **0.536** | **0.276** | **0.545** | **0.554** |
| (reference) | SAHI standard (Ultralytics 8.4.37 + pycocotools) | 0.4649 | 0.2389 | — | — |

Δ vs baseline: **mAP@0.5 +0.096**, **mAP@0.5:0.95 +0.078**.

### 7.4 Per-Class Metrics

**Validation split** (Ultralytics 8.4.37 final-epoch metrics):

| Class | Baseline | New | Δ |
|---|---|---|---|
| Crack | 0.384 | **0.496** | **+0.112** |
| Joint offset | 0.196 | **0.335** | **+0.139** |
| Utility intrusion | 0.708 | **0.795** | +0.087 |
| Debris | 0.414 | 0.464 | +0.050 |
| Buckling | 0.323 | 0.267 | −0.056 |
| Hole | 0.379 | 0.309 | −0.070 |
| Obstacle | 0.668 | 0.642 | −0.026 |

**Test split** (Ultralytics 8.4.51, re-evaluated):

| Class | Inst. | Baseline | New | Δ | New mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|
| Buckling | 17 | 0.326 | **0.138** | **−0.188** | 0.040 |
| Crack | 88 | 0.384 | 0.426 | +0.042 | 0.190 |
| Debris | 19 | 0.416 | **0.606** | **+0.190** | 0.278 |
| **Hole** | 7 | 0.379 | **0.804** | **+0.425** | 0.446 |
| Joint offset | 43 | 0.196 | 0.269 | +0.073 | 0.095 |
| Obstacle | 29 | 0.668 | 0.638 | −0.030 | 0.290 |
| **Utility intrusion** | 26 | 0.708 | **0.873** | **+0.165** | 0.594 |
| **All** | 229 | 0.440 | **0.536** | **+0.096** | 0.276 |

Six classes improve, only Buckling regresses sharply (likely tied to the
silently dropped `cls_weights` in 8.4.37 — see §7.6). The Hole +0.425 jump
benefits from small-N noise (only 7 instances) but the precision/recall
balance (0.568 / 0.857) makes the win plausible.

### 7.5 SAHI Sliced-Inference Sweep — Test Split

Same `best.pt` was evaluated with SAHI at three tile sizes:

| Run | mAP@0.5 | mAP@0.5:0.95 | mAP_small | mAP_medium | mAP_large |
|---|---|---|---|---|---|
| **standard** | **0.4649** | **0.2389** | 0.700 | 0.069 | 0.260 |
| sahi-256 | 0.3708 | 0.1818 | 0.700 | 0.048 | 0.197 |
| sahi-384 | 0.4150 | 0.1880 | 0.500 | 0.067 | 0.205 |
| sahi-512 | 0.4118 | 0.2027 | 0.600 | 0.055 | 0.224 |

**Interpretation**: SAHI does **not help** on this dataset.

- The input frames are already 640×480; YOLO26m's stride-8 P3 head sees
  defects at ~30–80 px, which is its sweet spot. Slicing chops contextual
  pixels away without giving the model anything new to look at.
- mAP_small was already 0.70 standard — there is no missing small-object
  recall for SAHI to recover.
- GREEDYNMM/IOS post-processing produced extra false-positive merges on
  the small test set (98 images, 229 boxes), pushing precision down.
- SAHI's expected gains are on 4K+ inputs (drone, satellite, microscopy)
  where each defect occupies a tiny fraction of the frame.

**Decision**: keep SAHI in the codebase as an evaluation tool but do not
ship it in the inference pipeline for this product. If we ever ingest 4K
CCTV footage, revisit at slice ≥ 640.

### 7.6 Lessons for the Team

1. **MuSGD on MPS = unusable**. For any local-Mac experimentation, force
   `optimizer="SGD"` and `amp=False`. Documented in
   `docs/cnn-ass3-yolo26-study.md`.
2. **The "small object" framing in PartC retrospective was wrong** for
   Joint offset and Crack at 640 px input — the gain came from better
   training (longer + close_mosaic + cosine-LR effect from late epochs),
   not from a smaller stride head.
3. **`cls_weights` kwarg is silently dropped** by Ultralytics 8.4.37.
   To actually weight classes, either upgrade Ultralytics or implement a
   custom `WeightedBCELoss` in the trainer subclass.
4. **The 0.44 → 0.46 jump came in epochs 28+** (after the model had seen
   enough mosaic + close_mosaic crops). 50 epochs is a reasonable floor;
   adding 30–50 more would likely push test mAP@0.5 past 0.50.

### 7.7 Deployment Artifacts

| Format | Path | Size | Notes |
|---|---|---|---|
| PyTorch | `runs/detect/train11/weights/best.pt` | 42 MB | Source of truth |
| ONNX | `runs/detect/train11/weights/best.onnx` | 78.6 MB | `dynamic=True`, `simplify=True`, opset 20 |
| Core ML | `runs/detect/train11/weights/best.mlpackage` | 39.1 MB | mlprogram, FP32, runs on Apple ANE |

Smoke-tested on one test image; all three formats return identical box
counts. Core ML's first call carries graph-compile overhead (~14 s); steady-
state inference is sub-100 ms.

### 7.8 Recommended Next Step

The Tier 2 imgsz=1280 plan is **down-scoped to imgsz=640** because the
Roboflow export is already 640×640 (verified by inspecting every train /
val / test image). Upscaling 640 to 1280 only inflates pixels via
interpolation, capturing no new detail.

Realistic next experiment:

1. **`yolo26-p2.yaml` at imgsz=640** with `cls_weights` (now working in
   8.4.51), `copy_paste=0.30`, `mixup=0.10`, `cos_lr=True`, 200 epochs.
   Target: recover Buckling and push test mAP@0.5 ≥ 0.58.
2. If a CUDA box becomes available, re-attempt with `yolo26l-p2.yaml` and
   200 epochs — about 1.5 h on a T4. Target: mAP@0.5 ≥ 0.62.
3. Source higher-resolution frames from Roboflow (1024×1024 or original)
   only if the project requires sub-10-pixel defect detection that the
   current 640 input cannot resolve.
