from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any

# Central logger for the whole experiment pipeline so all phases share one format.
LOGGER = logging.getLogger("yolo26-experiment")


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
    """Verify local runtime and pick CUDA when available."""
    import torch  # type: ignore[import-untyped]

    torch_version = _parse_version_tuple(torch.__version__)
    # Hard fail early so long training runs do not start on unsupported runtimes.
    if torch_version < min_torch:
        msg = (
            f"PyTorch {min_torch[0]}.{min_torch[1]}.{min_torch[2]}+ is required, "
            f"but found {torch.__version__}."
        )
        raise RuntimeError(msg)

    # Prefer CUDA automatically; caller decides final device argument format.
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    if device_type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        LOGGER.info("CUDA available: %s (VRAM %.1f GiB)", gpu_name, vram_gb)
    else:
        LOGGER.warning("CUDA not detected: running in CPU mode.")

    return device_type


def resolve_data_yaml(data_yaml: str) -> Path:
    """Resolve and validate dataset yaml path."""
    # Resolve to absolute path once so downstream logging is unambiguous.
    path = Path(data_yaml).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Could not find data.yaml path: {path}")
    return path


def load_model(weights: str) -> Any:
    """
    Load YOLO26 model from Ultralytics in Python runtime.

    yolo26m.pt must be available locally or via Ultralytics download.
    """
    from ultralytics import YOLO  # type: ignore[import-untyped]

    # Ultralytics handles local file loading and remote asset download behavior.
    return YOLO(weights)


def train_model(
    model: Any,
    data_yaml: Path,
    imgsz: int,
    epochs: int,
    batch: int,
    device: str | int,
    optimizer: str = "MuSGD",
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
        # Sewage-pipe-friendly augmentation/tuning (Ultralytics built-in)
        "hsv_v": 0.25,
        "hsv_s": 0.5,
        "fliplr": 0.5,
        "mosaic": 0.8,
        "patience": 30,
    }
    LOGGER.info("Training started: optimizer=%s, epochs=%s, batch=%s", optimizer, epochs, batch)

    # Try YOLO26-specific optimizer first when the runtime supports it.
    try:
        return model.train(**train_kwargs)
    except Exception as exc:
        if optimizer != "MuSGD":
            raise

        LOGGER.warning(
            "MuSGD failed (%s). Falling back to SGD because this Ultralytics version may not support it.",
            exc,
        )
        # Fallback keeps the run executable on environments where MuSGD is absent.
        train_kwargs["optimizer"] = "SGD"
        return model.train(**train_kwargs)


def validate_model(model: Any, data_yaml: Path, imgsz: int, device: str | int) -> Any:
    """Run quantitative validation and return metrics object."""
    # Returned metrics object includes mAP and other class-wise indicators.
    LOGGER.info("Validation started (check mAP@0.5 and mAP@0.5:0.95)")
    return model.val(data=str(data_yaml), imgsz=imgsz, device=device)


def export_tensorrt(
    model: Any,
    half: bool = True,
    dynamic: bool = True,
    workspace: int = 4,
) -> str:
    """Export model to TensorRT engine for high-speed local deployment."""
    # `half=True` generally gives the best latency/accuracy balance on RTX GPUs.
    LOGGER.info("TensorRT export started: half=%s, dynamic=%s, workspace=%sGiB", half, dynamic, workspace)
    export_result = model.export(
        format="engine",
        dynamic=dynamic,
        half=half,
        workspace=workspace,
    )
    return str(export_result)


def run_trt_inference(engine_path: str, test_image: str, device: str) -> Any:
    """Load exported TensorRT engine and run a sample inference."""
    from ultralytics import YOLO  # type: ignore[import-untyped]

    trt_model = YOLO(engine_path, task="detect")
    # Use this as a smoke test to verify the exported engine is usable.
    return trt_model.predict(test_image, device=device)


def build_arg_parser() -> argparse.ArgumentParser:
    # Keep all runtime knobs in CLI flags so experiments are reproducible.
    parser = argparse.ArgumentParser(description="YOLO26 sewage defect experiment pipeline")
    parser.add_argument(
        "--data",
        default="src/data/sewage-yolo26/data.yaml",
        help="Path to training/validation dataset yaml",
    )
    parser.add_argument("--weights", default="yolo26m.pt", help="Initial pretrained weights file (.pt)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--gpu-index", type=int, default=0, help="CUDA GPU index")
    parser.add_argument("--optimizer", default="MuSGD", help="Training optimizer")
    parser.add_argument("--workspace", type=int, default=4, help="TensorRT conversion workspace VRAM (GiB)")
    parser.add_argument("--test-image", default="", help="Test image path for TensorRT inference")
    parser.add_argument("--skip-export", action="store_true", help="Skip TensorRT export step")
    return parser


def main() -> None:
    """Run end-to-end YOLO26 local experiment pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    # Parse once at startup; every phase reads from the same immutable config.
    args = build_arg_parser().parse_args()

    # Phase 1: local environment + data setup
    LOGGER.info("Phase 1 - environment check and data path validation")
    device_type = verify_environment()
    data_yaml = resolve_data_yaml(args.data)
    # This call documents/validates the augmentation recipe used for custom datasets.
    _ = build_torch_augmentation()

    # Phase 2: model init
    LOGGER.info("Phase 2 - YOLO26 model initialization")
    model = load_model(args.weights)

    # Phase 3: training
    LOGGER.info("Phase 3 - training")
    # Ultralytics accepts either integer GPU index or "cpu".
    train_device: str | int = args.gpu_index if device_type == "cuda" else "cpu"
    _ = train_model(
        model=model,
        data_yaml=data_yaml,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=train_device,
        optimizer=args.optimizer,
    )

    # Phase 4: validation + TensorRT export/inference
    LOGGER.info("Phase 4 - validation and deployment")
    metrics = validate_model(model=model, data_yaml=data_yaml, imgsz=args.imgsz, device=train_device)
    LOGGER.info("Validation metrics object: %s", metrics)

    if args.skip_export:
        # Useful for quick iteration when only training/validation is needed.
        LOGGER.info("Skipping TensorRT export as requested.")
        return

    engine_path = export_tensorrt(model=model, half=True, dynamic=True, workspace=args.workspace)
    LOGGER.info("TensorRT engine created: %s", engine_path)

    if args.test_image:
        trt_device = f"cuda:{args.gpu_index}" if device_type == "cuda" else "cpu"
        results = run_trt_inference(engine_path=engine_path, test_image=args.test_image, device=trt_device)
        LOGGER.info("TensorRT inference completed: %s", results)
    else:
        LOGGER.info("No --test-image provided: skipping TensorRT inference test.")


if __name__ == "__main__":
    main()
