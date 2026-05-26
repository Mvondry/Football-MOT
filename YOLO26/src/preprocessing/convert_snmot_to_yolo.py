from __future__ import annotations

import argparse
import csv
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class MotBox:
    class_id: int  # 0-based YOLO class ID
    x: float
    y: float
    w: float
    h: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert SoccerNet-Tracking / MOT-style annotations to YOLO format. "
            "The script expects class-specific GT files such as gt_4_cls.txt."
        )
    )

    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory containing SoccerNet-Tracking sequence folders, e.g. data/tracking.",
    )

    parser.add_argument(
        "--train-split",
        type=str,
        required=True,
        help="Path to train split file.",
    )

    parser.add_argument(
        "--val-split",
        type=str,
        required=True,
        help="Path to validation split file.",
    )

    parser.add_argument(
        "--test-split",
        type=str,
        default=None,
        help="Optional path to test split file.",
    )

    parser.add_argument(
        "--gt-filename",
        type=str,
        default="gt_4_cls.txt",
        help="Name of the GT file inside each sequence gt/ directory.",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Output directory where YOLO dataset will be created.",
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=4,
        help="Number of YOLO classes.",
    )

    parser.add_argument(
        "--class-names",
        type=str,
        default="player,goalkeeper,ball,referee",
        help="Comma-separated class names in YOLO class order.",
    )

    parser.add_argument(
        "--transfer-mode",
        type=str,
        default="copy",
        choices=["copy", "hardlink", "symlink"],
        help="How images should be transferred to the YOLO dataset directory.",
    )

    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Optional limit of sequences per split for quick debugging.",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit of frames per sequence for quick debugging.",
    )

    return parser.parse_args()


def read_split_file(split_file: str | Path) -> List[Path]:
    split_file = Path(split_file)

    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    sequence_paths = []

    with open(split_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip().strip('"').strip("'")

            if not line:
                continue

            sequence_paths.append(Path(line))

    return sequence_paths


def resolve_sequence_path(data_root: str | Path, sequence_path: str | Path) -> Path:
    data_root = Path(data_root)
    sequence_path = Path(sequence_path)

    if sequence_path.is_absolute():
        return sequence_path

    direct_path = data_root / sequence_path

    if direct_path.exists():
        return direct_path

    # Fallback for split files containing only SNMOT-xxx instead of train/SNMOT-xxx.
    matches = [
        path for path in data_root.rglob(sequence_path.name)
        if path.is_dir() and path.name == sequence_path.name
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple folders named '{sequence_path.name}' found under {data_root}. "
            f"Use a more specific path in the split file, e.g. 'train/{sequence_path.name}'."
        )

    return direct_path


def sorted_images(image_dir: Path) -> List[Path]:
    image_paths = [
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS
    ]

    def sort_key(path: Path):
        return int(path.stem) if path.stem.isdigit() else path.stem

    return sorted(image_paths, key=sort_key)


def get_image_size(image_path: Path) -> Tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            width, height = image.size

        return width, height

    except Exception as exc:
        raise RuntimeError(f"Could not read image size from {image_path}") from exc


def transfer_image(src_path: Path, dst_path: Path, transfer_mode: str) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if dst_path.exists():
        return

    if transfer_mode == "symlink":
        os.symlink(str(src_path.resolve()), str(dst_path))
        return

    if transfer_mode == "hardlink":
        try:
            os.link(src_path, dst_path)
            return
        except OSError:
            shutil.copy2(src_path, dst_path)
            return

    shutil.copy2(src_path, dst_path)


def load_gt_by_frame(gt_file: Path, num_classes: int) -> Dict[int, List[MotBox]]:
    """
    Reads MOT-like GT file with class ID in the last column.

    Expected columns:
        frame_id, track_id, x, y, w, h, ..., class_id

    Class IDs in GT are expected to be 1-based.
    YOLO class IDs are 0-based.
    """
    if not gt_file.exists():
        raise FileNotFoundError(f"GT file not found: {gt_file}")

    boxes_by_frame: Dict[int, List[MotBox]] = {}

    with open(gt_file, "r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)

        for row in reader:
            if not row:
                continue

            row = [value.strip() for value in row]

            if len(row) < 11:
                continue

            try:
                frame_id = int(float(row[0]))
                x = float(row[2])
                y = float(row[3])
                w = float(row[4])
                h = float(row[5])
                class_id_1based = int(float(row[-1]))
            except ValueError:
                continue

            if w <= 0 or h <= 0:
                continue

            class_id_0based = class_id_1based - 1

            if not (0 <= class_id_0based < num_classes):
                continue

            boxes_by_frame.setdefault(frame_id, []).append(
                MotBox(
                    class_id=class_id_0based,
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                )
            )

    return boxes_by_frame


def convert_box_to_yolo(
    box: MotBox,
    image_width: int,
    image_height: int,
) -> Optional[Tuple[int, float, float, float, float]]:
    """
    Converts MOT box (x, y, w, h) in pixels to YOLO normalized format.

    The box is clipped to image boundaries before conversion.
    """
    x1 = max(0.0, box.x)
    y1 = max(0.0, box.y)
    x2 = min(float(image_width), box.x + box.w)
    y2 = min(float(image_height), box.y + box.h)

    clipped_w = x2 - x1
    clipped_h = y2 - y1

    if clipped_w <= 0 or clipped_h <= 0:
        return None

    x_center = (x1 + clipped_w / 2.0) / image_width
    y_center = (y1 + clipped_h / 2.0) / image_height
    width = clipped_w / image_width
    height = clipped_h / image_height

    return box.class_id, x_center, y_center, width, height


def write_yolo_label(
    label_path: Path,
    boxes: List[MotBox],
    image_width: int,
    image_height: int,
) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    for box in boxes:
        converted = convert_box_to_yolo(
            box=box,
            image_width=image_width,
            image_height=image_height,
        )

        if converted is None:
            continue

        class_id, x_center, y_center, width, height = converted

        lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    # Write empty label files as well. This explicitly marks images without objects.
    with open(label_path, "w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines) + "\n")


def write_dataset_yaml(
    output_root: Path,
    num_classes: int,
    class_names: List[str],
) -> Path:
    yaml_path = output_root / "snmot.yaml"
    root_str = str(output_root.resolve()).replace("\\", "/")

    lines = [
        f"path: {root_str}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {num_classes}",
        "names:",
    ]

    for class_id, class_name in enumerate(class_names):
        lines.append(f"  {class_id}: {class_name}")

    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return yaml_path


def convert_split(
    split_name: str,
    sequence_paths: List[Path],
    data_root: Path,
    output_root: Path,
    gt_filename: str,
    num_classes: int,
    transfer_mode: str,
    max_sequences: Optional[int] = None,
    max_frames: Optional[int] = None,
) -> None:
    if max_sequences is not None:
        sequence_paths = sequence_paths[:max_sequences]

    output_images_root = output_root / "images" / split_name
    output_labels_root = output_root / "labels" / split_name

    output_images_root.mkdir(parents=True, exist_ok=True)
    output_labels_root.mkdir(parents=True, exist_ok=True)

    for sequence_path in sequence_paths:
        sequence_dir = resolve_sequence_path(data_root, sequence_path)
        sequence_name = sequence_dir.name

        image_dir = sequence_dir / "img1"
        gt_file = sequence_dir / "gt" / gt_filename

        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        if not gt_file.exists():
            raise FileNotFoundError(f"GT file not found: {gt_file}")

        image_paths = sorted_images(image_dir)

        if max_frames is not None:
            image_paths = image_paths[:max_frames]

        if not image_paths:
            print(f"[WARNING] No images found in {image_dir}")
            continue

        boxes_by_frame = load_gt_by_frame(
            gt_file=gt_file,
            num_classes=num_classes,
        )

        for frame_index, source_image_path in enumerate(image_paths):
            if source_image_path.stem.isdigit():
                frame_id = int(source_image_path.stem)
            else:
                frame_id = frame_index + 1

            image_width, image_height = get_image_size(source_image_path)

            output_image_path = output_images_root / sequence_name / source_image_path.name
            output_label_path = output_labels_root / sequence_name / f"{source_image_path.stem}.txt"

            transfer_image(
                src_path=source_image_path,
                dst_path=output_image_path,
                transfer_mode=transfer_mode,
            )

            boxes = boxes_by_frame.get(frame_id, [])

            write_yolo_label(
                label_path=output_label_path,
                boxes=boxes,
                image_width=image_width,
                image_height=image_height,
            )

        print(f"[OK] {split_name}: {sequence_name} ({len(image_paths)} frames)")


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)

    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    class_names = [name.strip() for name in args.class_names.split(",") if name.strip()]

    if len(class_names) != args.num_classes:
        raise ValueError(
            f"Number of class names ({len(class_names)}) does not match "
            f"num_classes ({args.num_classes})."
        )

    output_root.mkdir(parents=True, exist_ok=True)

    train_sequences = read_split_file(args.train_split)
    val_sequences = read_split_file(args.val_split)
    test_sequences = read_split_file(args.test_split) if args.test_split else []

    convert_split(
        split_name="train",
        sequence_paths=train_sequences,
        data_root=data_root,
        output_root=output_root,
        gt_filename=args.gt_filename,
        num_classes=args.num_classes,
        transfer_mode=args.transfer_mode,
        max_sequences=args.max_sequences,
        max_frames=args.max_frames,
    )

    convert_split(
        split_name="val",
        sequence_paths=val_sequences,
        data_root=data_root,
        output_root=output_root,
        gt_filename=args.gt_filename,
        num_classes=args.num_classes,
        transfer_mode=args.transfer_mode,
        max_sequences=args.max_sequences,
        max_frames=args.max_frames,
    )

    if test_sequences:
        convert_split(
            split_name="test",
            sequence_paths=test_sequences,
            data_root=data_root,
            output_root=output_root,
            gt_filename=args.gt_filename,
            num_classes=args.num_classes,
            transfer_mode=args.transfer_mode,
            max_sequences=args.max_sequences,
            max_frames=args.max_frames,
        )
    else:
        (output_root / "images" / "test").mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / "test").mkdir(parents=True, exist_ok=True)

    yaml_path = write_dataset_yaml(
        output_root=output_root,
        num_classes=args.num_classes,
        class_names=class_names,
    )

    print()
    print("Conversion finished.")
    print(f"YOLO dataset root: {output_root}")
    print(f"Dataset YAML:      {yaml_path}")


if __name__ == "__main__":
    main()