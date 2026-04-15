"""Data extraction script.

Extracts images and YOLO labels from a dataset and saves them as `.pt` tensors:
- `feature/`: image tensors `[C, H, W]`
- `label/`: label tensors `[N, 5]` (`class, xc, yc, w, h`)
"""

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import cast

import torch
from PIL import Image
from torchvision import transforms

DEFAULT_BASE_DIR = Path(__file__).resolve().parent / "data" / "sewage-yolo26" / "train"
LOGGER = logging.getLogger(__name__)

# Converts PIL image -> Float Tensor [C, H, W] normalized to [0,1]
to_tensor = transforms.ToTensor()


def parse_args() -> Namespace:
    """Parse command line arguments for dataset extraction."""
    parser = ArgumentParser(description="Extract YOLO images/labels to .pt tensors.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Dataset split root directory (contains images/ and labels/).",
    )
    parser.add_argument(
        "--image-ext",
        type=str,
        default=".jpg",
        help="Image extension to process (e.g. .jpg, .png).",
    )
    parser.add_argument(
        "--inspect-count",
        type=int,
        default=3,
        help="How many extracted samples to inspect after extraction.",
    )
    parser.add_argument(
        "--skip-inspect",
        action="store_true",
        help="Skip sample inspection output after extraction.",
    )
    return parser.parse_args()


def load_image(img_path: Path) -> torch.Tensor:
    """Load image and return RGB tensor `[3, H, W]` in `[0.0, 1.0]`."""
    with Image.open(img_path) as img:
        return cast("torch.Tensor", to_tensor(img.convert("RGB")))


def load_label(txt_path: Path) -> torch.Tensor:
    """Load YOLO label file and return tensor `[N, 5]`; empty/missing -> `[0, 5]`."""
    rows: list[list[float]] = []
    if txt_path.exists():
        with txt_path.open("r", encoding="utf-8") as label_file:
            for line in label_file:
                row = line.strip()
                if row:
                    rows.append(list(map(float, row.split())))

    if rows:
        return torch.tensor(rows, dtype=torch.float32)
    return torch.zeros((0, 5), dtype=torch.float32)


def extract_all(
    *,
    img_dir: Path,
    label_dir: Path,
    feature_dir: Path,
    label_out: Path,
    image_ext: str,
) -> None:
    """Extract all matching images and labels and store them as tensors."""
    normalized_ext = image_ext.lower()
    img_files = sorted(
        [file_path for file_path in img_dir.iterdir() if file_path.suffix.lower() == normalized_ext]
    )

    LOGGER.info("Found %d images. Starting extraction...", len(img_files))
    LOGGER.info("")

    for img_path in img_files:
        stem = img_path.stem
        label_path = label_dir / f"{stem}.txt"
        feat_out = feature_dir / f"{stem}.pt"
        lbl_out = label_out / f"{stem}.pt"

        feature_tensor = load_image(img_path)
        torch.save(feature_tensor, feat_out)

        label_tensor = load_label(label_path)
        torch.save(label_tensor, lbl_out)

    LOGGER.info("Extraction completed!")
    LOGGER.info("  feature/ -> %s", feature_dir)
    LOGGER.info("  label/   -> %s", label_out)
    LOGGER.info("")


def inspect_samples(*, feature_dir: Path, label_out: Path, inspect_count: int) -> None:
    """Print a quick preview of extracted tensors."""
    if inspect_count <= 0:
        return

    feat_files = sorted(feature_dir.glob("*.pt"))[:inspect_count]

    LOGGER.info("=" * 60)
    LOGGER.info("Test: Inspect first %d samples (INPUT and LABEL)", len(feat_files))
    LOGGER.info("=" * 60)

    for i, feat_path in enumerate(feat_files, 1):
        stem = feat_path.stem
        label_path = label_out / f"{stem}.pt"

        feat = torch.load(feat_path, map_location="cpu")
        label = torch.load(label_path, map_location="cpu")

        _, height, width = feat.shape
        num_boxes = label.shape[0]

        input_preview: list[str] = []
        for row in range(min(1, height)):
            for col in range(min(3, width)):
                r = feat[0, row, col].item()
                g = feat[1, row, col].item()
                b = feat[2, row, col].item()
                input_preview.append(f"[RGB=({r:.3f},{g:.3f},{b:.3f}), W={col}, H={row}]")

        label_strs: list[str] = []
        for j in range(num_boxes):
            cls, xc, yc, box_w, box_h = label[j].tolist()
            label_strs.append(
                f"[class={int(cls)}, xc={xc:.4f}, yc={yc:.4f}, w={box_w:.4f}, h={box_h:.4f}]"
            )

        LOGGER.info("")
        LOGGER.info("-- Sample %d: %s --", i, stem)
        LOGGER.info("  Tensor shape: feature=%s (CxHxW)", list(feat.shape))
        LOGGER.info("  dtype:        %s", feat.dtype)
        LOGGER.info("  Pixel range:  [%.4f, %.4f]", feat.min(), feat.max())
        LOGGER.info("")
        LOGGER.info("  INPUT (top-left 3 pixels preview):")
        for preview in input_preview:
            LOGGER.info("    %s", preview)
        LOGGER.info("  ... total pixels: %d", width * height)
        LOGGER.info("")
        LOGGER.info("  LABEL (%d bounding boxes):", num_boxes)
        if label_strs:
            for label_str in label_strs:
                LOGGER.info("    %s", label_str)
        else:
            LOGGER.info("    (no objects)")

    LOGGER.info("")
    LOGGER.info("%s", "=" * 60)


def main() -> None:
    """Script entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    base_dir = args.base_dir.resolve()

    img_dir = base_dir / "images"
    label_dir = base_dir / "labels"
    feature_dir = base_dir / "feature"
    label_out = base_dir / "label"

    if not img_dir.exists():
        msg = f"Image directory does not exist: {img_dir}"
        raise FileNotFoundError(msg)
    if not label_dir.exists():
        msg = f"Label directory does not exist: {label_dir}"
        raise FileNotFoundError(msg)

    feature_dir.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    extract_all(
        img_dir=img_dir,
        label_dir=label_dir,
        feature_dir=feature_dir,
        label_out=label_out,
        image_ext=args.image_ext,
    )
    if not args.skip_inspect:
        inspect_samples(
            feature_dir=feature_dir,
            label_out=label_out,
            inspect_count=args.inspect_count,
        )


if __name__ == "__main__":
    main()
