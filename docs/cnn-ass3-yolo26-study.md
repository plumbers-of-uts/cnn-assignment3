# YOLO26 Study Notes — for Sewage Defect Detection

Companion to `cnn-ass3-PartC-improvement-plan.md`. Captures what we learned
about YOLO26 architecture, optimizer, and Apple MPS pitfalls while debugging
the local training pipeline.

---

## 1. Architecture Overview

Source: `ultralytics/cfg/models/26/yolo26.yaml` (Ultralytics 8.4.37).

### Backbone

```
0  Conv 64,3,2          # P1/2
1  Conv 128,3,2         # P2/4
2  C3k2 256, False, 0.25
3  Conv 256,3,2         # P3/8
4  C3k2 512, False, 0.25
5  Conv 512,3,2         # P4/16
6  C3k2 512, True
7  Conv 1024,3,2        # P5/32
8  C3k2 1024, True
9  SPPF 1024, 5, 3, True
10 C2PSA 1024           # ← new: cross-stage Partial Self-Attention block
```

### Head (P3/P4/P5 default)

Standard FPN-PAN: upsample → concat with backbone P4/P3 → downsample +
concat → 3 Detect heads at strides 8, 16, 32.

### Scaling

| scale | params  | GFLOPs | depth | width | max_ch |
|:-----:|:-------:|:------:|:-----:|:-----:|:------:|
| n     | 2.57 M  | 6.1    | 0.50  | 0.25  | 1024   |
| s     | 10.01 M | 22.8   | 0.50  | 0.50  | 1024   |
| **m** | **21.90 M** | **75.4** | **0.50** | **1.00** | **512** |
| l     | 26.30 M | 93.8   | 1.00  | 1.00  | 512    |
| x     | 58.99 M | 209.5  | 1.00  | 1.50  | 512    |

The PartC baseline uses `m` (21.9 M params). For local MPS training we kept
`m` but reduced `batch=4`, `imgsz=640` to fit the memory budget.

### Key new modules vs YOLOv8

| Module | What it does | Why it matters |
|---|---|---|
| `C3k2` | C3 block with kernel size 2 in the bottleneck — lighter than C2f | More channels at same compute → better feature mixing |
| `C2PSA` | Cross-Stage Partial Self-Attention | Adds global context; pairs well with small-object detection |
| `end2end=True` | Built-in NMS-free head (one box per object) | Removes NMS hyperparameters; predictions are already filtered |
| `reg_max: 1` | DFL bins reduced from 16 (YOLOv8) → 1 | Faster regression head, slight accuracy trade-off |

### P2 / P6 variants

- `yolo26-p2.yaml` adds a stride-4 detection head (P2/4) → 4 detect outputs.
  Adds ≈3.5 GFLOPs at `n` scale, but dramatically improves small-object
  recall. The PartC weak classes (Crack, Hole, Joint offset) sit at < 30 px
  on 640 inputs, which is exactly the P2/4 sweet spot.
- `yolo26-p6.yaml` adds an extra stride-64 head for very large objects.
  Not relevant for pipe defects.

---

## 2. Default Training Recipe

```python
optimizer = "MuSGD"  # automatic fallback to AdamW when iterations <= 10k
lr0        = 0.01    # SGD-style; MuSGD inherits the same
momentum   = 0.937
weight_decay = 0.0005
mosaic     = 0.8
close_mosaic = 10    # disable mosaic in last 10 epochs
imgsz      = 640
epochs     = 200     # PartC team set 85 actual → early-stopped at 57
```

Important defaults:
- `amp=True` — automatic mixed precision (FP16). Works on CUDA, **fragile on MPS**.
- `cls_weights` — newer Ultralytics versions accept this; ours (8.4.37) does not.

---

## 3. MuSGD Optimizer — Why It Stalled on MPS

`MuSGD` lives in `ultralytics/optim/muon.py`. It is a hybrid:

```python
class MuSGD(optim.Optimizer):
    # 2D / 4D conv weights → Muon update (orthogonalized gradient)
    # Other params (bias, BN) → vanilla SGD
```

The Muon path:

```python
def zeropower_via_newtonschulz5(G):
    X = G.bfloat16()                  # ← bf16 cast
    X /= X.norm() + eps
    for _ in range(5):                # 5 iterations
        A = X @ X.T                   # ← bf16 matmul on backbone weights
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X
```

**MPS problem**: bfloat16 matmul lacks a fast MPS kernel for the shapes
produced by YOLO26's conv weights. PyTorch silently falls back to CPU
(`PYTORCH_ENABLE_MPS_FALLBACK=1` makes this implicit fallback succeed
without crashing). Each backward pass therefore moves backbone gradients
host ↔ device twice and runs the orthogonalization on CPU. The observed
cost was ~580 s/batch — about 100x slower than SGD on the same device.

### Workaround we adopted

```python
OPTIMIZER = "SGD"        # No bf16 matmul → no fallback
AMP_ENABLED = False      # MPS AMP for YOLO26 also falls back
MPS_BATCH_SIZE = 4
workers = 0              # MPS dataloader is single-threaded
```

Measured cost on M1 Pro 16 GiB, yolo26m, imgsz=640, batch=4: **~2.5 s/batch**
once warmed up. 172 batches/epoch × 2.5 s ≈ 7 min/epoch + ~30 s validation,
i.e. ~50 epochs in 6–7 hours.

When CUDA is available we switch back to MuSGD because that is where the
upstream defaults were tuned (and where bf16 matmul has fast kernels).

---

## 4. Why YOLO26 Helps the Sewage Task

| Sewage-task pain | YOLO26 mitigation |
|---|---|
| Cracks span only a few pixels at 640² | C2PSA gives long-range attention; P2 head adds stride-4 features |
| Dataset is tiny (686 train images) | end2end + reg_max=1 reduce hyperparameter sensitivity |
| Class imbalance 22:1 (Crack vs Hole) | Built-in copy-paste and mosaic; CIW exposed via `cls_weights` |
| Inference speed matters for CCTV | reg_max=1 + NMS-free head → fewer postprocess ops |

The architectural fit is genuine — the bottleneck for our run is mostly the
optimizer/MPS issue, not the model itself.

---

## 5. Pitfalls We Hit (Run Log)

1. **MuSGD + MPS = ~580 s/batch.** Symptom: `1364 s/it` on the very first
   batch, slow recovery to ~580 s. Mitigation: switch to SGD on MPS.
2. **AMP + MPS for YOLO26.** Symptom: NaN losses and additional CPU
   fallbacks. Mitigation: `amp=False` on MPS only.
3. **`cls_weights` arg.** Ultralytics 8.4.37 rejects this kwarg silently.
   We catch the `TypeError` and retry without it; the import lives in
   `src/initial_experiments.py::train_model`.
4. **`workers > 0` on MPS.** The dataloader spawns subprocesses that each
   try to acquire MPS, leading to crashes. Force `workers=0` on MPS.
5. **YOLO26 in colab vs. local.** PartC baseline used Tesla T4 with default
   MuSGD/AMP and finished in ≈1 h. The same config on M1 Pro is unusable.
   The performance is hardware-bound for now, not algorithmic.

---

## 6. References

- Hidayatullah & Tubagus (2026), *YOLO26: A comprehensive architecture
  overview and key improvements*. arXiv:2602.14582.
- Ultralytics docs: `https://docs.ultralytics.com/models/yolo26`
- MuSGD paper inspiration: *Muon: A Momentum Update from Orthogonalization*
  (Newton-Schulz quintic iteration).
- Source files browsed locally:
  - `ultralytics/cfg/models/26/yolo26.yaml`
  - `ultralytics/cfg/models/26/yolo26-p2.yaml`
  - `ultralytics/optim/muon.py`
  - `ultralytics/engine/trainer.py::build_optimizer`
