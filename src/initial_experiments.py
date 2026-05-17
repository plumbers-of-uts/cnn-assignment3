from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# Central logger for the whole experiment pipeline so all phases share one format.
LOGGER = logging.getLogger("yolo26-experiment")

# ---------------------------------------------------------------------------
# Experiment configuration (edit these constants instead of passing CLI args).
# ---------------------------------------------------------------------------
DATA_YAML_PATH = "src/data/sewage-yolo26/data.yaml"
# Baseline pretrained checkpoint path used for transfer learning.
WEIGHTS_PATH = "src/model/yolo26m.pt"

# Training hyperparameters
EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 32
# Lower batch size for Apple MPS to reduce out-of-memory risk.
# MuSGD lacks MPS kernels (silent CPU fallback → ~580s/batch), so we use SGD
# below; batch=4 keeps per-step memory tractable on M1 Pro 16 GiB.
MPS_BATCH_SIZE = 4
# Lower batch size for CPU to avoid excessive RAM pressure and long step times.
CPU_BATCH_SIZE = 8
GPU_INDEX = 0
# MuSGD on MPS triggers CPU fallback for most ops (~580s/batch). Use SGD on
# MPS/CPU and switch back to MuSGD only when CUDA is available.
OPTIMIZER = "SGD"
# Disable AMP on MPS — mixed-precision kernels for YOLO26 are not fully
# implemented yet and cause further fallbacks. Keep AMP on for CUDA.
AMP_ENABLED = False

# ---------------------------------------------------------------------------
# Tier 2 (high-resolution + P2 head + aggressive augmentation) — targets the
# small/thin defect classes (Crack, Hole, Joint offset) that drag baseline mAP.
# Enable for the boosted retraining run; keep disabled for the baseline.
# ---------------------------------------------------------------------------
TIER2_ENABLE = False
TIER2_MODEL_YAML = "yolo26-p2.yaml"  # Adds stride-4 P2 head for small objects.
TIER2_IMAGE_SIZE = 1280
TIER2_EPOCHS = 300
TIER2_PATIENCE = 50
TIER2_BATCH_SIZE = 8  # imgsz=1280 quadruples VRAM vs 640 — lower batch by default.
TIER2_AUGMENT_OVERRIDES: dict[str, float] = {
    "mosaic": 1.0,
    "close_mosaic": 20,
    "mixup": 0.10,
    "copy_paste": 0.30,
    "scale": 0.6,
    "degrees": 10.0,
    "translate": 0.10,
    "hsv_v": 0.3,
    "hsv_s": 0.6,
    "fliplr": 0.5,
}
TIER2_COSINE_LR = True

# Checkpoint behavior
# When enabled, training attempts to continue from RESUME_CHECKPOINT_PATH.
# Keep this True to automatically recover from interrupted runs.
RESUME_TRAINING = True
RESUME_CHECKPOINT_PATH = "runs/detect/train/weights/last.pt"

# Long-tail imbalance controls
# Enable this to print explicit reminders about YOLO26 long-tail behaviors
# (e.g., STAL/ProgLoss) during training startup.
ENABLE_LONG_TAIL_LOGGING = True

# Optional class importance weights for severe class imbalance.
# Keep this mapping aligned with class names in data.yaml.
# Example:
# {
#     "Buckling": 1.0,
#     "Crack": 2.5,
#     "Debris": 1.0,
#     "Hole": 2.0,
#     "Joint offset": 1.4,
#     "Obstacle": 1.2,
#     "Utility intrusion": 2.8,
# }
# Default class weighting for sewage long-tail imbalance.
# Frequent/less-severe classes remain near 1.0, while rare/high-impact defects
# get larger weights to improve recall during optimization.
CLASS_IMPORTANCE_WEIGHTS: dict[str, float] | None = {
    "Buckling": 1.8,
    "Crack": 2.6,
    "Debris": 1.0,
    "Hole": 2.4,
    "Joint offset": 1.7,
    "Obstacle": 1.2,
    "Utility intrusion": 2.8,
}
USE_CLASS_IMPORTANCE_WEIGHTS = True

# Export and inference behavior
SKIP_EXPORT = False
TENSORRT_WORKSPACE_GIB = 4
EXPORT_DYNAMIC = True
# INT8 is enabled by default to maximize throughput and memory efficiency.
# Keep FP16 off by default when INT8 is enabled to avoid mixed-intent config.
EXPORT_HALF = False
EXPORT_INT8 = True
TEST_IMAGE_PATH = ""  # Leave empty to skip TensorRT inference smoke test.

# .pt dataset sanity-check behavior (for outputs from extract_data.py)
ENABLE_PT_SANITY_CHECK = True
PT_SANITY_SPLIT = "train"
# Use 0 to scan all files in the split.
PT_SANITY_MAX_SAMPLES = 200
PT_LABEL_NDIM = 2
PT_LABEL_COLUMNS = 5
PT_TOP_CLASS_COUNT = 5


def _parse_version_tuple(version_text: str) -> tuple[int, int, int]:
    """Extract semantic version tuple from arbitrary version string."""
    # Accept strings like "2.3.1+cu121" and only keep the numeric core.
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", version_text)
    if not match:
        return (0, 0, 0)
    return tuple(int(part) for part in match.groups())


def build_torch_augmentation() -> Any:
    """
    Build PyTorch-level augmentation pipeline for dark sewage-pipe conditions.

    Note:
        This is an example transform pipeline that can complement
        Ultralytics built-in augmentation in train().
        Connect this directly to a custom Dataset if needed.
    """
    from torchvision import transforms  # type: ignore[import-untyped]

    # Keep this light-weight: this pipeline is intended as a practical baseline
    # for low-light sewer inspections, not a full augmentation search space.
    return transforms.Compose(
        [
            transforms.ColorJitter(brightness=0.35, contrast=0.4),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    )


def verify_environment(min_torch: tuple[int, int, int] = (1, 8, 0)) -> str:
    """Verify local runtime and pick the best available accelerator."""
    import torch  # type: ignore[import-untyped]

    torch_version = _parse_version_tuple(torch.__version__)
    # Hard fail early so long training runs do not start on unsupported runtimes.
    if torch_version < min_torch:
        msg = (
            f"PyTorch {min_torch[0]}.{min_torch[1]}.{min_torch[2]}+ is required, "
            f"but found {torch.__version__}."
        )
        raise RuntimeError(msg)

    # Prefer CUDA first, then Apple Metal (MPS), and finally CPU.
    if torch.cuda.is_available():
        device_type = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        LOGGER.info("CUDA available: %s (VRAM %.1f GiB)", gpu_name, vram_gb)
    elif torch.backends.mps.is_available():
        device_type = "mps"
        LOGGER.info("Apple Metal (MPS) detected: running on M1/M2 GPU backend.")
    else:
        device_type = "cpu"
        LOGGER.warning("CUDA not detected: running in CPU mode.")

    return device_type


def resolve_data_yaml(data_yaml: str) -> Path:
    """Resolve and validate dataset yaml path."""
    # Resolve to absolute path once so downstream logging is unambiguous.
    path = Path(data_yaml).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Could not find data.yaml path: {path}")
    return path


def resolve_resume_checkpoint_path(explicit_checkpoint_path: str) -> Path | None:
    """
    Resolve a usable resume checkpoint path.

    Resolution order:
    1) Explicit path in config, when it exists.
    2) Most recently modified `runs/detect/*/weights/last.pt`.
    3) None when no candidate exists.
    """
    explicit_path = Path(explicit_checkpoint_path).resolve()
    if explicit_path.exists():
        return explicit_path

    runs_root = Path("runs/detect").resolve()
    if not runs_root.exists():
        return None

    candidates = sorted(
        runs_root.glob("*/weights/last.pt"),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_model(
    weights: str,
    resume_training: bool = False,
    resume_checkpoint_path: Path | None = None,
    architecture_yaml: str | None = None,
) -> Any:
    """
    Load YOLO26 model from Ultralytics in Python runtime.

    yolo26m.pt must be available locally or via Ultralytics download.

    architecture_yaml:
        When provided, builds the model from this Ultralytics yaml (e.g.,
        "yolo26-p2.yaml" to add a P2 small-object head) and loads pretrained
        weights on top via `.load()`. Ignored when resume_training=True.
    """
    from ultralytics import YOLO  # type: ignore[import-untyped]

    # If resume mode is enabled, load the checkpoint artifact directly.
    # Otherwise use the baseline pretrained weights.
    if resume_training:
        checkpoint = resume_checkpoint_path
        if checkpoint is None or not checkpoint.exists():
            raise FileNotFoundError("Resume mode enabled but no valid checkpoint was resolved.")
        LOGGER.info("Loading model from checkpoint: %s", checkpoint)
        return YOLO(str(checkpoint))

    if architecture_yaml:
        # Build the alternate architecture (e.g., +P2 head) then transfer
        # compatible pretrained weights from `weights` for faster convergence.
        LOGGER.info(
            "Loading architecture from yaml=%s with pretrained transfer from %s",
            architecture_yaml,
            weights,
        )
        model = YOLO(architecture_yaml)
        try:
            model.load(weights)
        except Exception as exc:  # noqa: BLE001 — transfer is best-effort.
            LOGGER.warning(
                "Pretrained transfer from %s failed (%s); continuing from scratch.",
                weights,
                exc,
            )
        return model

    # Ultralytics handles local file loading and remote asset download behavior.
    LOGGER.info("Loading baseline weights: %s", weights)
    return YOLO(weights)


def _extract_yaml_class_names(data_yaml: Path) -> list[str]:
    """
    Read class names from data.yaml.

    This helper isolates yaml parsing so imbalance logic remains transparent.
    """
    import yaml  # type: ignore[import-untyped]

    with data_yaml.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}

    names = payload.get("names", [])
    if isinstance(names, dict):
        # Handle dict-form names: {0: "class_a", 1: "class_b", ...}
        return [str(names[idx]) for idx in sorted(names)]
    if isinstance(names, list):
        return [str(name) for name in names]
    return []


def build_class_importance_vector(
    data_yaml: Path,
    class_importance_weights: dict[str, float] | None,
) -> list[float] | None:
    """
    Build per-class weight vector aligned with data.yaml class order.

    Returns None when weights are disabled or not provided.
    """
    if not USE_CLASS_IMPORTANCE_WEIGHTS or not class_importance_weights:
        return None

    class_names = _extract_yaml_class_names(data_yaml)
    if not class_names:
        LOGGER.warning(
            "Could not resolve class names from data.yaml; class importance weights are skipped.",
        )
        return None

    # Default to weight=1.0 for any class omitted by user config.
    vector = [float(class_importance_weights.get(name, 1.0)) for name in class_names]
    LOGGER.info("Class importance weights enabled: %s", dict(zip(class_names, vector, strict=False)))
    return vector


def _as_float(value: Any) -> float | None:
    """Safely convert scalar-like values to float."""
    if value is None:
        return None
    try:
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)
    except (TypeError, ValueError):
        return None


def _best_f1_score(f1_values: Any) -> float | None:
    """
    Extract a single representative F1 score from Ultralytics outputs.

    Ultralytics may expose class-wise arrays/tensors. For imbalance-heavy
    datasets, the best F1 across confidence thresholds is often practical to log.
    """
    if f1_values is None:
        return None

    if hasattr(f1_values, "tolist"):
        f1_values = f1_values.tolist()

    if isinstance(f1_values, (list, tuple)):
        stack = list(f1_values)
        flattened: list[float] = []
        while stack:
            item = stack.pop()
            if isinstance(item, (list, tuple)):
                stack.extend(item)
                continue
            scalar = _as_float(item)
            if scalar is not None:
                flattened.append(scalar)
        return max(flattened) if flattened else None

    return _as_float(f1_values)


def _format_metric(value: float | None) -> str:
    """Format optional metric value for stable console logging."""
    return f"{value:.4f}" if value is not None else "N/A"


def _is_mps_oom_error(exc: BaseException) -> bool:
    """Return True when exception message indicates MPS out-of-memory."""
    text = str(exc).lower()
    return "mps backend out of memory" in text or ("out of memory" in text and "mps" in text)


def log_domain_metrics(metrics: Any) -> None:
    """
    Log sewage-domain-friendly metrics in a compact, comparable format.

    The goal is to track key detection quality metrics across runs:
    - mAP@0.5
    - mAP@0.5:0.95
    - F1-score (best extracted value)
    """
    box_metrics = getattr(metrics, "box", None)
    map50 = _as_float(getattr(box_metrics, "map50", None))
    map50_95 = _as_float(getattr(box_metrics, "map", None))
    f1_best = _best_f1_score(getattr(box_metrics, "f1", None))

    LOGGER.info(
        "Domain metrics | mAP@0.5=%s | mAP@0.5:0.95=%s | F1(best)=%s",
        _format_metric(map50),
        _format_metric(map50_95),
        _format_metric(f1_best),
    )


def train_model(
    model: Any,
    data_yaml: Path,
    imgsz: int,
    epochs: int,
    batch: int,
    device: str | int,
    optimizer: str = "MuSGD",
    class_importance_vector: list[float] | None = None,
    resume_training: bool = False,
    patience: int = 30,
    extra_train_overrides: dict[str, Any] | None = None,
    *,
    cosine_lr: bool = False,
) -> Any:
    """Run YOLO26 training with fallback for unsupported optimizer."""
    # The default settings are tuned for the requested 640 training setup and
    # dark/low-contrast sewer imagery.
    train_kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "optimizer": optimizer,
        "plots": True,
        "resume": resume_training,
        "amp": AMP_ENABLED,
        "workers": 0 if device == "mps" else 8,
        # Sewage-pipe-friendly augmentation/tuning (Ultralytics built-in)
        "hsv_v": 0.25,
        "hsv_s": 0.5,
        "fliplr": 0.5,
        "mosaic": 0.8,
        "patience": patience,
        "cos_lr": cosine_lr,
    }
    if extra_train_overrides:
        # Tier 2 augmentation/scheduling overrides go through here.
        train_kwargs.update(extra_train_overrides)
    if class_importance_vector is not None:
        # Some Ultralytics versions may not expose class-weight args publicly.
        # We still try to pass them, then gracefully fallback if unsupported.
        train_kwargs["cls_weights"] = class_importance_vector

    if ENABLE_LONG_TAIL_LOGGING:
        LOGGER.info(
            "Long-tail strategy check: YOLO26 relies on built-in STAL/ProgLoss behavior; "
            "custom CIW is %s.",
            "enabled" if class_importance_vector is not None else "disabled",
        )

    LOGGER.info(
        "Training started: optimizer=%s, epochs=%s, batch=%s, resume=%s",
        optimizer,
        epochs,
        batch,
        resume_training,
    )

    # Try YOLO26-specific optimizer first when the runtime supports it.
    active_kwargs = dict(train_kwargs)
    while True:
        try:
            return model.train(**active_kwargs)
        except Exception as exc:
            # If class-importance argument is rejected by this runtime, retry
            # without it to keep experiments runnable.
            if "cls_weights" in active_kwargs:
                LOGGER.warning(
                    "Class importance weights were not accepted by the current trainer (%s). "
                    "Retrying without cls_weights.",
                    exc,
                )
                active_kwargs.pop("cls_weights", None)
                continue

            # If MPS memory is exhausted, reduce batch and retry automatically.
            if active_kwargs.get("device") == "mps" and _is_mps_oom_error(exc):
                current_batch = int(active_kwargs["batch"])
                if current_batch <= 1:
                    raise
                next_batch = max(1, current_batch // 2)
                LOGGER.warning(
                    "MPS out of memory at batch=%s (%s). Retrying with batch=%s.",
                    current_batch,
                    exc,
                    next_batch,
                )
                active_kwargs["batch"] = next_batch
                if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
                continue

            if optimizer != "MuSGD":
                raise

            LOGGER.warning(
                "MuSGD failed (%s). Falling back to SGD because this Ultralytics version may not support it.",
                exc,
            )
            # Fallback keeps the run executable on environments where MuSGD is absent.
            active_kwargs["optimizer"] = "SGD"
            return model.train(**active_kwargs)


def validate_model(model: Any, data_yaml: Path, imgsz: int, device: str | int) -> Any:
    """Run quantitative validation and return metrics object."""
    # Returned metrics object includes mAP and other class-wise indicators.
    LOGGER.info("Validation started (check mAP@0.5 and mAP@0.5:0.95)")
    return model.val(data=str(data_yaml), imgsz=imgsz, device=device)


def export_tensorrt(
    model: Any,
    half: bool = True,
    int8: bool = False,
    dynamic: bool = True,
    workspace: int = 4,
) -> str:
    """Export model to TensorRT engine for high-speed local deployment."""
    # INT8 and FP16 are both quantization modes. If both are enabled, prefer INT8
    # and disable FP16 explicitly to avoid ambiguous export intent.
    if int8 and half:
        LOGGER.warning("Both INT8 and FP16 were enabled. Prioritizing INT8 and disabling FP16.")
        half = False

    LOGGER.info(
        "TensorRT export started: half=%s, int8=%s, dynamic=%s, workspace=%sGiB",
        half,
        int8,
        dynamic,
        workspace,
    )
    export_result = model.export(
        format="engine",
        dynamic=dynamic,
        half=half,
        int8=int8,
        workspace=workspace,
    )
    return str(export_result)


def run_trt_inference(engine_path: str, test_image: str, device: str) -> Any:
    """Load exported TensorRT engine and run a sample inference."""
    from ultralytics import YOLO  # type: ignore[import-untyped]

    trt_model = YOLO(engine_path, task="detect")
    # Use this as a smoke test to verify the exported engine is usable.
    return trt_model.predict(test_image, device=device)


@dataclass
class PtBBoxStats:
    """Aggregate bbox stats for extracted label tensors."""

    total_boxes: int = 0
    width_sum: float = 0.0
    height_sum: float = 0.0
    width_min: float | None = None
    width_max: float | None = None
    height_min: float | None = None
    height_max: float | None = None

    def update(self, label_tensor: torch.Tensor, class_counter: Counter[int]) -> None:
        """Update class and bbox summary stats from one valid label tensor."""
        boxes_in_file = int(label_tensor.shape[0])
        self.total_boxes += boxes_in_file

        for cls_value in label_tensor[:, 0].tolist():
            class_counter[int(cls_value)] += 1

        for box_w, box_h in label_tensor[:, 3:5].tolist():
            box_w_float = float(box_w)
            box_h_float = float(box_h)

            self.width_sum += box_w_float
            self.height_sum += box_h_float

            self.width_min = (
                box_w_float if self.width_min is None else min(self.width_min, box_w_float)
            )
            self.width_max = (
                box_w_float if self.width_max is None else max(self.width_max, box_w_float)
            )
            self.height_min = (
                box_h_float if self.height_min is None else min(self.height_min, box_h_float)
            )
            self.height_max = (
                box_h_float if self.height_max is None else max(self.height_max, box_h_float)
            )


def _resolve_pt_dirs(data_yaml: Path, split: str) -> tuple[Path, Path] | None:
    """Resolve feature/label directories for extracted pt tensors."""
    split_root = data_yaml.parent / split
    feature_dir = split_root / "feature"
    label_dir = split_root / "label"

    if not label_dir.exists():
        LOGGER.warning("PT sanity-check skipped: label directory not found: %s", label_dir)
        return None
    if not feature_dir.exists():
        LOGGER.warning("PT sanity-check skipped: feature directory not found: %s", feature_dir)
        return None
    return feature_dir, label_dir


def run_pt_sanity_check(data_yaml: Path, split: str, max_samples: int) -> None:
    """Inspect extracted `.pt` labels and log dataset health before training."""
    resolved_dirs = _resolve_pt_dirs(data_yaml=data_yaml, split=split)
    if resolved_dirs is None:
        return

    feature_dir, label_dir = resolved_dirs

    label_files = sorted(label_dir.glob("*.pt"))
    if not label_files:
        LOGGER.warning("PT sanity-check skipped: no label .pt files found in %s", label_dir)
        return

    if max_samples > 0:
        label_files = label_files[:max_samples]

    total_files = len(label_files)
    malformed_count = 0
    empty_label_count = 0
    missing_feature_count = 0
    bbox_stats = PtBBoxStats()
    class_counter: Counter[int] = Counter()

    for label_path in label_files:
        feature_path = feature_dir / label_path.name
        if not feature_path.exists():
            missing_feature_count += 1

        label_tensor = torch.load(label_path, map_location="cpu")
        if label_tensor.ndim != PT_LABEL_NDIM or label_tensor.shape[1] != PT_LABEL_COLUMNS:
            malformed_count += 1
            continue

        if int(label_tensor.shape[0]) == 0:
            empty_label_count += 1
            continue

        bbox_stats.update(label_tensor=label_tensor, class_counter=class_counter)

    mean_boxes_per_file = bbox_stats.total_boxes / total_files if total_files else 0.0
    mean_width = bbox_stats.width_sum / bbox_stats.total_boxes if bbox_stats.total_boxes else 0.0
    mean_height = bbox_stats.height_sum / bbox_stats.total_boxes if bbox_stats.total_boxes else 0.0
    top_classes = class_counter.most_common(PT_TOP_CLASS_COUNT)

    LOGGER.info(
        "PT sanity-check (%s) | files=%s | malformed=%s | empty_labels=%s | missing_features=%s",
        split,
        total_files,
        malformed_count,
        empty_label_count,
        missing_feature_count,
    )
    LOGGER.info(
        "PT sanity-check (%s) | total_boxes=%s | mean_boxes/file=%.2f | top_classes=%s",
        split,
        bbox_stats.total_boxes,
        mean_boxes_per_file,
        top_classes,
    )
    LOGGER.info(
        "PT sanity-check (%s) | bbox_w(min/mean/max)=%.4f/%.4f/%.4f | "
        "bbox_h(min/mean/max)=%.4f/%.4f/%.4f",
        split,
        bbox_stats.width_min if bbox_stats.width_min is not None else 0.0,
        mean_width,
        bbox_stats.width_max if bbox_stats.width_max is not None else 0.0,
        bbox_stats.height_min if bbox_stats.height_min is not None else 0.0,
        mean_height,
        bbox_stats.height_max if bbox_stats.height_max is not None else 0.0,
    )


def main() -> None:
    """Run end-to-end YOLO26 local experiment pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    # Phase 1: local environment + data setup
    LOGGER.info("Phase 1 - environment check and data path validation")
    device_type = verify_environment()
    data_yaml = resolve_data_yaml(DATA_YAML_PATH)
    # This call documents/validates the augmentation recipe used for custom datasets.
    _ = build_torch_augmentation()
    if ENABLE_PT_SANITY_CHECK:
        run_pt_sanity_check(
            data_yaml=data_yaml,
            split=PT_SANITY_SPLIT,
            max_samples=PT_SANITY_MAX_SAMPLES,
        )

    resume_checkpoint: Path | None = None
    should_resume = RESUME_TRAINING
    if RESUME_TRAINING:
        resume_checkpoint = resolve_resume_checkpoint_path(RESUME_CHECKPOINT_PATH)
        if resume_checkpoint is None:
            should_resume = False
            LOGGER.warning(
                "RESUME_TRAINING=True but no last.pt checkpoint was found. "
                "Falling back to a fresh run from baseline weights.",
            )
        else:
            LOGGER.info("Resume checkpoint resolved: %s", resume_checkpoint)

    # Phase 2: model init
    LOGGER.info("Phase 2 - YOLO26 model initialization")
    architecture_yaml = TIER2_MODEL_YAML if (TIER2_ENABLE and not should_resume) else None
    model = load_model(
        WEIGHTS_PATH,
        resume_training=should_resume,
        resume_checkpoint_path=resume_checkpoint,
        architecture_yaml=architecture_yaml,
    )

    # Phase 3: training
    LOGGER.info("Phase 3 - training (tier2=%s)", TIER2_ENABLE)
    # Ultralytics accepts integer GPU index, "mps", or "cpu".
    train_device: str | int = GPU_INDEX if device_type == "cuda" else device_type
    baseline_batch = TIER2_BATCH_SIZE if TIER2_ENABLE else BATCH_SIZE
    if device_type == "cuda":
        train_batch_size = baseline_batch
    elif device_type == "mps":
        train_batch_size = min(MPS_BATCH_SIZE, baseline_batch)
        LOGGER.info(
            "MPS mode detected: overriding batch size from %s to %s.",
            baseline_batch,
            train_batch_size,
        )
    else:
        train_batch_size = min(CPU_BATCH_SIZE, baseline_batch)
    if device_type == "cpu":
        LOGGER.info(
            "CPU mode detected: overriding batch size from %s to %s.",
            baseline_batch,
            train_batch_size,
        )
    class_importance_vector = build_class_importance_vector(
        data_yaml=data_yaml,
        class_importance_weights=CLASS_IMPORTANCE_WEIGHTS,
    )
    active_imgsz = TIER2_IMAGE_SIZE if TIER2_ENABLE else IMAGE_SIZE
    active_epochs = TIER2_EPOCHS if TIER2_ENABLE else EPOCHS
    active_patience = TIER2_PATIENCE if TIER2_ENABLE else 30
    active_augment_overrides = TIER2_AUGMENT_OVERRIDES if TIER2_ENABLE else None
    active_cosine_lr = TIER2_COSINE_LR if TIER2_ENABLE else False
    _ = train_model(
        model=model,
        data_yaml=data_yaml,
        imgsz=active_imgsz,
        epochs=active_epochs,
        batch=train_batch_size,
        device=train_device,
        optimizer=OPTIMIZER,
        class_importance_vector=class_importance_vector,
        resume_training=should_resume,
        patience=active_patience,
        extra_train_overrides=active_augment_overrides,
        cosine_lr=active_cosine_lr,
    )

    # Phase 4: validation + TensorRT export/inference
    LOGGER.info("Phase 4 - validation and deployment")
    metrics = validate_model(model=model, data_yaml=data_yaml, imgsz=active_imgsz, device=train_device)
    log_domain_metrics(metrics)
    LOGGER.info("Validation metrics object: %s", metrics)

    if SKIP_EXPORT:
        # Useful for quick iteration when only training/validation is needed.
        LOGGER.info("Skipping TensorRT export as requested.")
        return

    engine_path = export_tensorrt(
        model=model,
        half=EXPORT_HALF,
        int8=EXPORT_INT8,
        dynamic=EXPORT_DYNAMIC,
        workspace=TENSORRT_WORKSPACE_GIB,
    )
    LOGGER.info("TensorRT engine created: %s", engine_path)

    if TEST_IMAGE_PATH:
        trt_device = f"cuda:{GPU_INDEX}" if device_type == "cuda" else "cpu"
        results = run_trt_inference(
            engine_path=engine_path,
            test_image=TEST_IMAGE_PATH,
            device=trt_device,
        )
        LOGGER.info("TensorRT inference completed: %s", results)
    else:
        LOGGER.info("TEST_IMAGE_PATH is empty: skipping TensorRT inference test.")


if __name__ == "__main__":
    main()
