from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch
import ultralytics
import yaml
from ultralytics import YOLO, settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


CLASS_NAMES = {
    0: "player",
    1: "goalkeeper",
    2: "ball",
    3: "referee",
    4: "other",
}


CLASS_COLORS_BGR = {
    "player": (255, 0, 0),        # blue
    "goalkeeper": (0, 0, 255),    # red
    "ball": (255, 255, 0),        # cyan
    "referee": (0, 255, 255),     # yellow
    "other": (128, 128, 128),     # gray
    "unknown": (255, 255, 255),   # white
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize YOLO detections on images or on a selected dataset split."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML experiment config.",
    )

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained YOLO weights.",
    )

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Optional path to dataset YAML. Overrides value from config.",
    )

    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help=(
            "Optional image or directory to visualize. "
            "If not set, images are loaded from the selected split in the dataset YAML."
        ),
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split used when --source is not provided.",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="results",
        help="Root directory for visualization outputs.",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=10,
        help="Maximum number of images to visualize.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for predictions.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Inference image size. If not set, value from config is used.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use, e.g. '0' or 'cpu'. If not set, value from config is used.",
    )

    return parser.parse_args()


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset_yaml(data_yaml_path: str | Path) -> dict:
    data_yaml_path = Path(data_yaml_path)

    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML file not found: {data_yaml_path}")

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def resolve_data_yaml(config: dict, data_override: str | None) -> Path:
    data_yaml = Path(data_override) if data_override is not None else Path(config["data"])

    if not data_yaml.is_absolute():
        data_yaml = resolve_project_path(data_yaml)

    return data_yaml


def resolve_source_from_dataset(data_yaml_path: Path, split: str) -> Path:
    dataset_config = load_dataset_yaml(data_yaml_path)

    dataset_root = Path(dataset_config["path"])
    split_path = Path(dataset_config[split])

    if split_path.is_absolute():
        return split_path

    return dataset_root / split_path


def collect_images(source_path: Path, max_images: int | None) -> list[Path]:
    if not source_path.exists():
        raise FileNotFoundError(f"Source does not exist: {source_path}")

    if source_path.is_file():
        image_paths = [source_path]
    else:
        image_paths = [
            path for path in source_path.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        def sort_key(path: Path):
            return str(path.parent), int(path.stem) if path.stem.isdigit() else path.stem

        image_paths = sorted(image_paths, key=sort_key)

    if max_images is not None:
        image_paths = image_paths[:max_images]

    return image_paths


def get_relative_output_path(image_path: Path, source_root: Path) -> Path:
    if source_root.is_file():
        return Path(image_path.name)

    try:
        return image_path.relative_to(source_root)
    except ValueError:
        return Path(image_path.name)


def get_text_color(background_color):
    if background_color in [(0, 255, 255), (255, 255, 0)]:
        return (0, 0, 0)

    return (255, 255, 255)


def draw_box_with_label(image, x1, y1, x2, y2, label, class_name):
    height, width = image.shape[:2]

    x1 = int(max(0, min(x1, width - 1)))
    y1 = int(max(0, min(y1, height - 1)))
    x2 = int(max(0, min(x2, width - 1)))
    y2 = int(max(0, min(y2, height - 1)))

    color = CLASS_COLORS_BGR.get(class_name, CLASS_COLORS_BGR["unknown"])

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness,
    )

    text_x1 = x1
    text_y1 = max(y1 - text_height - baseline - 4, 0)
    text_x2 = min(x1 + text_width + 6, width - 1)
    text_y2 = min(text_y1 + text_height + baseline + 6, height - 1)

    cv2.rectangle(
        image,
        (text_x1, text_y1),
        (text_x2, text_y2),
        color,
        -1,
    )

    text_color = get_text_color(color)

    cv2.putText(
        image,
        label,
        (text_x1 + 3, text_y2 - baseline - 3),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def draw_yolo_result(image, result):
    if result.boxes is None or len(result.boxes) == 0:
        return image

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)

    for box, score, class_id in zip(boxes, scores, classes):
        class_name = CLASS_NAMES.get(int(class_id), "unknown")
        label = f"{class_name} {float(score):.2f}"

        x1, y1, x2, y2 = box

        draw_box_with_label(
            image=image,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            label=label,
            class_name=class_name,
        )

    return image


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    experiment_name = config["experiment_name"]

    output_root = resolve_project_path(args.output_root)
    output_dir = output_root / experiment_name / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = resolve_project_path(weights_path)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file does not exist: {weights_path}")

    data_yaml = resolve_data_yaml(config, args.data)

    if args.source is not None:
        source_path = Path(args.source)
        if not source_path.is_absolute():
            source_path = resolve_project_path(source_path)
    else:
        source_path = resolve_source_from_dataset(
            data_yaml_path=data_yaml,
            split=args.split,
        )

    training_config = config.get("training", {})
    evaluation_config = config.get("evaluation", {})

    imgsz = args.imgsz
    if imgsz is None:
        imgsz = evaluation_config.get("imgsz", training_config.get("imgsz", 640))

    device = args.device
    if device is None:
        device = evaluation_config.get("device", training_config.get("device", "0"))

    settings.update(
        {
            "runs_dir": str(output_root),
            "wandb": False,
        }
    )

    print("=== YOLO VISUALIZATION START ===")
    print(f"ultralytics: {ultralytics.__version__}")
    print(f"torch      : {torch.__version__}")
    print(f"cuda avail : {torch.cuda.is_available()}")
    print()
    print(f"experiment : {experiment_name}")
    print(f"weights    : {weights_path}")
    print(f"data       : {data_yaml}")
    print(f"source     : {source_path}")
    print(f"output dir : {output_dir}")
    print(f"device     : {device}")
    print(f"imgsz      : {imgsz}")
    print(f"conf       : {args.conf}")
    print("===============================")

    image_paths = collect_images(
        source_path=source_path,
        max_images=args.max_images,
    )

    if not image_paths:
        raise RuntimeError(f"No images found in source: {source_path}")

    model = YOLO(str(weights_path))
    saved_count = 0

    for image_path in image_paths:
        results = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=args.conf,
            device=device,
            verbose=False,
        )

        result = results[0]

        visualized_image = cv2.imread(str(image_path))

        if visualized_image is None:
            print(f"[WARNING] Could not read image: {image_path}")
            continue

        visualized_image = draw_yolo_result(
            image=visualized_image,
            result=result,
        )

        relative_output_path = get_relative_output_path(
            image_path=image_path,
            source_root=source_path,
        )

        save_path = output_dir / relative_output_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(save_path), visualized_image)

        print(f"Saved: {save_path}")
        saved_count += 1

    print()
    print(f"Visualization finished. Saved images: {saved_count}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()