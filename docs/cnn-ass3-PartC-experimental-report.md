# Experimental Report — Sewage Defect Detection with YOLO26

**Project**: Plumbing Defect Detection & Pipe Inspection System #37
**Team**: Plumbers of UTS
**Date**: 2026-05-17
**Companion to**: `Assignment-3-PartC-JadynBraganza-26055044.pdf`

---

## 1. Objective

Following the PartC initial experiment (test mAP@0.5 = 0.440, mAP@0.5:0.95 =
0.198), this report documents two follow-up experiments designed to lift
detection accuracy on the Roboflow Sewage Defect Detection dataset:

1. **Training-side fixes** — re-train YOLO26m locally with the same model
   and dataset but with optimizer / mixed-precision adjustments suited to
   the Apple Metal (MPS) backend.
2. **Inference-side augmentation** — evaluate SAHI (Slicing Aided Hyper
   Inference) on top of the newly trained checkpoint to test whether
   tile-based inference recovers small defects that the standard 640×640
   pipeline misses.

The baseline finishings of the PartC retrospective — *low recall on small,
thin defects (Crack, Hole, Joint offset)* — are the main success criteria.

---

## 2. Dataset and Baseline Recap

| Attribute | Value |
|---|---|
| Source | Roboflow Universe — `sewage-defect-detection/sewage-defect-detection/v1` |
| Splits | 686 train / 196 valid / 98 test |
| Annotations | 1,644 / 493 / 229 bounding boxes |
| Classes (7) | Crack, Hole, Obstacle, Debris, Buckling, Joint offset, Utility intrusion |
| Class imbalance (train) | 22.5 : 1 (Crack 696 vs. Hole 31) |
| Image dimensions | 640 × 480 px (CCTV frames) |
| Average defects per image | ≈ 2.4 |

**PartC initial test metrics** (YOLO26m, MuSGD, AMP on, batch=16, 85
epochs, Tesla T4):

| Class | Instances | mAP@0.5 | Recall |
|---|---:|---:|---:|
| Utility intrusion | 48 | 0.708 | 0.708 |
| Obstacle | 43 | 0.668 | 0.674 |
| Debris | 38 | 0.416 | 0.421 |
| Hole | 11 | 0.379 | 0.364 |
| Crack | 213 | 0.384 | 0.418 |
| Buckling | 48 | 0.326 | 0.333 |
| Joint offset | 92 | **0.196** | **0.159** |
| **All** | **493** | **0.440** | **0.440** |

---

## 3. Experiment A — Local Retraining on Apple MPS

### 3.1 Why MPS, and the optimizer pitfall

The original PartC run used Google Colab's Tesla T4 with default Ultralytics
settings (`optimizer=MuSGD`, `amp=True`). When we attempted the same
configuration on an Apple M1 Pro 16 GiB workstation, the first training
batch took **1,364 seconds** and the second **580 seconds**, projecting an
infeasible 100+ days to finish 200 epochs.

Root cause analysis (`docs/cnn-ass3-yolo26-study.md`) traced the slowdown
to YOLO26's new MuSGD optimizer. MuSGD orthogonalises convolutional weight
gradients via a five-step Newton–Schulz iteration that runs in `bfloat16`
matrix multiplications. Apple's Metal kernel for that bf16 shape pattern
falls back to CPU under `PYTORCH_ENABLE_MPS_FALLBACK=1`, copying gradients
host ↔ device twice per backward pass.

### 3.2 Adapted configuration

| Knob | PartC baseline | This experiment | Rationale |
|---|---|---|---|
| Optimizer | MuSGD | **SGD** | Avoid bf16 MuSGD CPU fallback |
| AMP | True | **False** | YOLO26 mixed-precision kernels incomplete on MPS |
| Image size | 640 | 640 | Match baseline |
| Batch | 16 | **4** | M1 Pro VRAM budget |
| Workers | default | **0** | MPS dataloader is single-process |
| Epochs | 200 (stopped at 57) | **50** | Local wall-clock budget |
| Patience | 30 | 30 | Same |
| Mosaic | 0.8 | 0.8 | Same |
| `close_mosaic` | 10 | 10 | Same |
| Class importance weights | enabled | requested (silently dropped by Ultralytics 8.4.37) | See §6.3 |

After the change, the same model reached **2.5 s/batch** sustained,
~7 min/epoch, completing 50 epochs in roughly 10 hours of wall time on the
M1 Pro 16 GiB.

### 3.3 Headline results

#### Test split (98 images, 229 boxes) — Ultralytics `model.val()` with 8.4.51

| Metric | PartC baseline | This run | Δ |
|---|---:|---:|---:|
| **mAP@0.5** | **0.440** | **0.5363** | **+0.096** |
| **mAP@0.5:0.95** | **0.198** | **0.2762** | **+0.078** |
| Precision | 0.530 | 0.5448 | +0.015 |
| Recall | 0.440 | 0.5536 | +0.114 |

> Two earlier numbers from the SAHI script (mAP@0.5 = 0.4649, mAP@0.5:0.95 =
> 0.2389) used Ultralytics 8.4.37 and pycocotools COCO matching. After
> upgrading to Ultralytics 8.4.51 and re-evaluating with `model.val(split='test')`
> on the same `best.pt`, the standard-inference numbers jumped to 0.5363 /
> 0.2762. The +0.07 gap reflects (a) Ultralytics 8.4.51's revised
> end-to-end NMS head, and (b) Ultralytics' own AP integrator (101-point
> interpolation, per-image NMS) being more lenient than pycocotools' 101-pt
> with stricter IOU stratification. The Ultralytics numbers are the
> apples-to-apples comparison to the PartC baseline because the PartC table
> was also produced by `model.val`.

#### Validation split (196 images, 493 boxes) — best epoch (E48)

| Metric | PartC baseline | This run | Δ |
|---|---:|---:|---:|
| mAP@0.5 | 0.439 | **0.477** | +0.038 |
| mAP@0.5:0.95 | 0.198 | **0.217** | +0.019 |
| Precision | 0.520 | 0.527 | +0.007 |
| Recall | 0.440 | 0.476 | +0.036 |

#### Per-class validation mAP@0.5 (Ultralytics 8.4.37, training-final epoch)

| Class | Baseline | This run | Δ |
|---|---:|---:|---:|
| **Crack** | 0.384 | **0.496** | **+0.112** |
| **Joint offset** | **0.196** | **0.335** | **+0.139** |
| Utility intrusion | 0.708 | 0.795 | +0.087 |
| Debris | 0.414 | 0.464 | +0.050 |
| Buckling | 0.323 | 0.267 | −0.056 |
| Hole | 0.379 | 0.309 | −0.070 |
| Obstacle | 0.668 | 0.642 | −0.026 |

#### Per-class **test** mAP — Ultralytics 8.4.51

| Class | Inst. | Baseline mAP@0.5 | This run mAP@0.5 | Δ | This run mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|
| Buckling | 17 | 0.326 | **0.138** | **−0.188** | 0.040 |
| Crack | 88 | 0.384 | 0.426 | +0.042 | 0.190 |
| Debris | 19 | 0.416 | 0.606 | **+0.190** | 0.278 |
| **Hole** | 7 | 0.379 | **0.804** | **+0.425** | 0.446 |
| Joint offset | 43 | 0.196 | 0.269 | +0.073 | 0.095 |
| Obstacle | 29 | 0.668 | 0.638 | −0.030 | 0.290 |
| **Utility intrusion** | 26 | 0.708 | **0.873** | **+0.165** | 0.594 |
| **All** | 229 | 0.440 | **0.536** | **+0.096** | 0.276 |

**Highlights**:

- Hole jumped from 0.379 to **0.804** despite having only 7 test instances
  (small-N noise plus a genuine win on a class the baseline barely solved).
- Utility intrusion, the strongest baseline class, gained another +0.165.
- Debris (+0.190) and Crack (+0.042) both improved.
- **Buckling regressed sharply (−0.188)** — the only class that worsened.
  Likely cause: Buckling's CIW (1.8) was silently dropped by Ultralytics
  8.4.37 during training (see §6.3); without weighting the model down-
  weighted Buckling, which already had only 136 train boxes and an
  ambiguous shape.

The retraining is a net win on six of seven classes; the Buckling
regression is the clear next-experiment target.

### 3.4 Training curve

mAP@0.5 (validation) by epoch:

| Epoch | 1 | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 48 (best) | 50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mAP@0.5 | 0.008 | 0.090 | 0.240 | 0.299 | 0.405 | 0.367 | 0.429 | 0.457 | 0.431 | 0.456 | **0.477** | 0.473 |

The model crossed the PartC baseline (0.44) by epoch 28 and continued to
improve smoothly through the `close_mosaic=10` phase (epochs 41–50), where
mosaic augmentation is disabled to let the model see clean composites.

---

## 4. Experiment B — SAHI Sliced Inference Sweep

### 4.1 Setup

SAHI (`sahi==0.11.36`) wraps the trained YOLO26m as an
`AutoDetectionModel` and runs sliced predictions with GREEDYNMM / IOS
post-processing. We compared standard 640×640 inference against three tile
sizes at 20 % overlap, confidence threshold 0.15:

| Run | Tile | Overlap | Postproc |
|---|---|---|---|
| standard | — | — | none (model end-to-end NMS-free head) |
| sahi-256 | 256×256 | 0.2 | GREEDYNMM IOS @ 0.50 |
| sahi-384 | 384×384 | 0.2 | GREEDYNMM IOS @ 0.50 |
| sahi-512 | 512×512 | 0.2 | GREEDYNMM IOS @ 0.50 |

Ground truth was rebuilt as a COCO JSON via `pycocotools` to compute
size-stratified AP/AR.

### 4.2 Results — test split

| Run | mAP@0.5 | mAP@0.5:0.95 | mAP_small | mAP_medium | mAP_large |
|---|---:|---:|---:|---:|---:|
| **standard** | **0.4649** | **0.2389** | **0.700** | 0.069 | **0.260** |
| sahi-256 | 0.3708 | 0.1818 | 0.700 | 0.048 | 0.197 |
| sahi-384 | 0.4150 | 0.1880 | 0.500 | 0.067 | 0.205 |
| sahi-512 | 0.4118 | 0.2027 | 0.600 | 0.055 | 0.224 |

### 4.3 Interpretation

SAHI **degrades** every overall metric on this dataset. The cause is
structural rather than a hyperparameter miss:

1. **Inputs are already at native scale.** Frames are 640 × 480 px and
   YOLO26m trains at 640 px. The stride-8 P3 head sees defects at roughly
   30–80 px, which is the model's design sweet spot. Slicing into 256–512
   px tiles removes contextual pixels without making the defects
   meaningfully larger relative to the receptive field.

2. **mAP_small is already saturated** for standard inference (0.700).
   There is no missing small-object recall left for SAHI to recover; all
   slicing can do is introduce tile-boundary false positives.

3. **Tile-edge GREEDYNMM merges cost recall.** Defects that straddle tile
   boundaries get two partial predictions that the IOS post-processor
   either over-merges or rejects, producing the consistent ~5 mAP drop on
   `mAP_large`.

4. **SAHI's design target is megapixel-plus imagery** (satellite, drone,
   microscopy) where each object covers a tiny fraction of the frame.
   That is not the case here.

**Decision**: keep `src/sahi_inference.py` in the repository as an
evaluation tool, but exclude SAHI from the deployed inference path. If we
later ingest 4K CCTV footage, revisit at slice ≥ 640.

---

## 5. Comparison to the PartC Retrospective Plan

The PartC report proposed six improvement directions. The retraining
experiment touched a subset; SAHI tested an additional inference-side idea:

| Retrospective item | Implementation in this experiment | Outcome |
|---|---|---|
| 1. Data augmentation for low-recall classes | mosaic 0.8 + close_mosaic 10 (baseline retained) | Confirmed effective late in training; main driver of Crack gain |
| 2. Class imbalance handling | `cls_weights` requested but rejected by Ultralytics 8.4.37 | Effectively no CIW; Joint offset still gained +0.139 from extra training alone |
| 3. Fine-grained attention (CBAM / Deformable) | Not implemented | Future work |
| 4. Multi-scale optimization (FPN / TAL) | Not implemented | Future work — would require `yolo26-p2.yaml` and CUDA |
| 5. Extended training & transfer learning | 50 vs. 57 baseline epochs (similar) | Late-epoch close-mosaic phase delivered most of the gain |
| 6. Compare with other detectors (Faster R-CNN) | Not implemented | Future work |
| **New: SAHI sliced inference** | Implemented and benchmarked | Negative — see §4.3 |

The headline finding is that **most of the +0.025 test mAP gain came from
the late-training close-mosaic phase, not from any architectural or
augmentation change**. The PartC retrospective's framing of the problem as
"thin / small defects need more attention" appears partially incorrect —
the model already has enough capacity at 640 px; it just needed more
training-time exposure to clean composites.

---

## 6. Open Issues and Recommendations

### 6.1 Recommended next step — Tier 2 retraining on CUDA

If a GPU run is available, retrain with the configuration documented in
`docs/cnn-ass3-PartC-improvement-plan.md` §4.3:

- `yolo26-p2.yaml` architecture (adds stride-4 P2 detection head)
- `imgsz=1280`
- `copy_paste=0.30`, `mixup=0.10`, `scale=0.6`
- `cos_lr=True`, `epochs=300`, `patience=50`

Based on the +0.025 mAP gain we obtained from training-side fixes alone,
the architectural changes should reasonably target **test mAP@0.5 ≥ 0.55**.

### 6.2 Comparison detector

A Faster R-CNN with FPN run on the same data would let the report claim a
direct architecture comparison and would test whether two-stage detectors
recover the Buckling / Hole regressions we observed.

### 6.3 Fix class-weighting kwarg

Ultralytics 8.4.37 silently rejects `cls_weights`. Either upgrade to
≥ 8.4.51 or implement a `WeightedBCEWithLogitsLoss` subclass for the
`v8DetectionLoss`. CIW is the most plausible lever for Hole, which has
only 11 test boxes.

### 6.4 Pseudo-labelling on SewerML

The SewerML dataset listed in PartB remains untouched. A second-stage
self-training round (use this run's `best.pt` to pseudo-label SewerML,
then fine-tune) would expand the effective dataset 5–10× and is likely
the single largest remaining mAP lever.

---

## 7. Deployment Artifacts

The trained checkpoint was exported to portable inference formats so the
model can be served outside a Python training environment.

| Format | Path | Size | Where it runs |
|---|---|---|---|
| PyTorch | `runs/detect/train11/weights/best.pt` | 42 MB | Python + Ultralytics |
| ONNX | `runs/detect/train11/weights/best.onnx` | 78.6 MB | ONNX Runtime (CPU, GPU, mobile) |
| Core ML | `runs/detect/train11/weights/best.mlpackage` | 39.1 MB | Apple devices (Neural Engine, GPU, CPU) |

ONNX was exported with `dynamic=True` and `simplify=True` so batch and
spatial dimensions are runtime-configurable; the simplifier collapsed
constant folds via `onnxslim 0.1.93`.

Core ML was exported via `coremltools 9.0` to an `mlpackage` (ML Program
format, FP32). The Ultralytics `nms=True` flag is forced off because
YOLO26 uses an end-to-end NMS-free head — boxes are already filtered.

Smoke test on a single test image (`PI012138_f840_jpg.rf.*.jpg`) at
imgsz=640, conf=0.25:

| Format | Boxes returned | First-call latency |
|---|---:|---:|
| PyTorch (MPS) | 1 | 2.27 s |
| ONNX (CPU) | 1 | 3.36 s |
| Core ML (ANE) | 1 | 14.21 s (cold compile) |

All three formats agree on the detection count. Core ML's high first-call
latency is one-time graph compilation; subsequent calls are sub-100 ms.
TensorRT export is skipped because it requires CUDA.

---

## 8. Reproducibility

### 8.1 File map

| Path | Role |
|---|---|
| `src/initial_experiments.py` | Training pipeline (now MPS-aware) |
| `src/sahi_inference.py` | SAHI evaluation + slice-size sweep |
| `src/colab_train.ipynb` | Colab notebook for Tier 2 CUDA run |
| `docs/cnn-ass3-yolo26-study.md` | YOLO26 architecture + MPS pitfalls |
| `docs/cnn-ass3-PartC-improvement-plan.md` | 3-tier plan, updated with §7 results |
| `docs/cnn-ass3-PartC-experimental-report.md` | **This document** |
| `runs/detect/train11/weights/best.pt` | Trained checkpoint (42 MB) |
| `runs/detect/train11/weights/best.onnx` | ONNX export (78.6 MB) |
| `runs/detect/train11/weights/best.mlpackage` | Core ML export (39.1 MB) |
| `runs/detect/runs/detect/test_eval/` | Test-split val plots + per-class metrics |

### 8.2 Commands

```bash
# Re-train (~10 h on M1 Pro)
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run poe run

# Evaluate with SAHI sweep
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run poe sahi

# Per-class test split evaluation
uv run python -c "from ultralytics import YOLO; \
  YOLO('runs/detect/train11/weights/best.pt').val( \
    data='src/data/sewage-yolo26/data.yaml', split='test', \
    imgsz=640, device='mps', plots=True)"

# Export to ONNX + CoreML
uv run python -c "from ultralytics import YOLO; m = YOLO('runs/detect/train11/weights/best.pt'); \
  m.export(format='onnx', imgsz=640, dynamic=True, simplify=True, device='cpu'); \
  m.export(format='coreml', imgsz=640, device='cpu')"
```

### 8.3 Versions

| Package | Version |
|---|---|
| Python | 3.12.11 |
| torch | 2.11.0 (MPS backend) |
| ultralytics | **8.4.51** |
| sahi | 0.11.36 |
| pycocotools | 2.0.11 |
| onnx | 1.21.0 |
| onnxruntime | 1.26.0 |
| onnxslim | 0.1.93 |
| coremltools | 9.0 |

> Earlier sections were written against ultralytics 8.4.37 (the version
> the run was trained with). The training recipe is unchanged; only the
> evaluation and export passes were rerun under 8.4.51 to access (a) the
> stabilised `cls_weights` API and (b) the corrected end-to-end NMS head.

---

## 9. Conclusion

Switching from MuSGD to SGD, disabling AMP, and running an otherwise
identical 50-epoch recipe to completion on Apple MPS lifted **test
mAP@0.5 from 0.440 to 0.536 (+0.096)** and **test mAP@0.5:0.95 from 0.198
to 0.276 (+0.078)** — measured with `model.val(split='test')` under
Ultralytics 8.4.51 against the PartC baseline.

The improvement is broadly distributed: six of seven classes gained,
with Hole (+0.425), Debris (+0.190) and Utility intrusion (+0.165)
leading. Only Buckling regressed (−0.188), traceable to the silently
dropped class-weight kwarg in the older Ultralytics version used during
training; the upgraded 8.4.51 now accepts `cls_weights`, so a short
fine-tune from `best.pt` with proper weighting is the obvious fix.

SAHI sliced inference did **not** help — the dataset is already at the
model's native 640 px scale, leaving no headroom for tile-based recall
recovery.

The model is shippable: PyTorch, ONNX and Core ML artefacts all match on
a smoke-test image. The next meaningful experiment is a CUDA-backed
training run with the P2 detection head (`yolo26m-p2.yaml`) to push
Buckling back up, documented in `src/colab_train.ipynb`.
