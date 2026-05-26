from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch
import torchvision
import yaml
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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
        description="Visualize Faster R-CNN detections on SoccerNet-Tracking frames."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML experiment config.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to Faster R-CNN checkpoint.",
    )

    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory containing SoccerNet-Tracking sequences.",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="results",
        help="Root directory for visualization outputs.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Split from config used for visualization.",
    )

    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help=(
            "Optional image, image directory, or sequence directory. "
            "If not set, images are loaded from the selected split."
        ),
    )

    parser.add_argument(
        "--max-sequences",
        type=int,
        default=3,
        help="Maximum number of sequences to visualize when --source is not provided.",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=5,
        help="Maximum number of frames per sequence to visualize.",
    )

    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Detection score threshold.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use, e.g. 'cuda' or 'cpu'.",
    )

    return parser.parse_args()


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def read_split_file(split_file: str | Path) -> list[Path]:
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

    matches = [
        path
        for path in data_root.rglob(sequence_path.name)
        if path.is_dir() and path.name == sequence_path.name
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple folders named '{sequence_path.name}' found under {data_root}. "
            f"Use a more specific path in the split file, e.g. 'train/{sequence_path.name}'."
        )

    raise FileNotFoundError(
        f"Sequence directory not found: {sequence_path} under {data_root}"
    )


def list_images(image_dir: str | Path) -> list[Path]:
    image_dir = Path(image_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = [
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    def sort_key(path: Path):
        return int(path.stem) if path.stem.isdigit() else path.stem

    return sorted(image_paths, key=sort_key)


def create_faster_rcnn_model(num_classes: int):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes,
    )

    return model


def load_checkpoint(model, checkpoint_path: str | Path, device: torch.device):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        str(checkpoint_path),
        map_location=device,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    return model


def normalize_class_name_for_color(class_name: str) -> str:
    class_name = str(class_name).lower()

    if "player" in class_name:
        return "player"

    if "goalkeeper" in class_name:
        return "goalkeeper"

    if "ball" in class_name:
        return "ball"

    if "referee" in class_name:
        return "referee"

    if "other" in class_name:
        return "other"

    return "unknown"


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

    color_key = normalize_class_name_for_color(class_name)
    color = CLASS_COLORS_BGR.get(color_key, CLASS_COLORS_BGR["unknown"])

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


def draw_faster_rcnn_predictions(image, prediction, class_names: list[str], score_threshold: float):
    boxes = prediction["boxes"].detach().cpu().numpy()
    scores = prediction["scores"].detach().cpu().numpy()
    labels = prediction["labels"].detach().cpu().numpy().astype(int)

    for box, score, label_id in zip(boxes, scores, labels):
        score = float(score)
        label_id = int(label_id)

        if score < score_threshold:
            continue

        if label_id == 0:
            continue

        if 0 <= label_id < len(class_names):
            class_name = class_names[label_id]
        else:
            class_name = "unknown"

        label = f"{class_name} {score:.2f}"

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


def image_to_tensor(frame_bgr, device: torch.device):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(frame_rgb).to(device)
    image_tensor = image_tensor.permute(2, 0, 1).float() / 255.0

    return image_tensor


def collect_source_images(source_path: Path, max_frames: int | None) -> list[Path]:
    if not source_path.exists():
        raise FileNotFoundError(f"Source does not exist: {source_path}")

    if source_path.is_file():
        return [source_path]

    if (source_path / "img1").exists():
        image_paths = list_images(source_path / "img1")
    else:
        image_paths = [
            path for path in source_path.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        def sort_key(path: Path):
            return str(path.parent), int(path.stem) if path.stem.isdigit() else path.stem

        image_paths = sorted(image_paths, key=sort_key)

    if max_frames is not None:
        image_paths = image_paths[:max_frames]

    return image_paths


def save_visualized_image(
    model,
    image_path: Path,
    output_path: Path,
    class_names: list[str],
    score_threshold: float,
    device: torch.device,
) -> bool:
    frame = cv2.imread(str(image_path))

    if frame is None:
        print(f"[WARNING] Could not read image: {image_path}")
        return False

    image_tensor = image_to_tensor(frame, device)

    with torch.no_grad():
        prediction = model([image_tensor])[0]

    visualized_image = frame.copy()

    draw_faster_rcnn_predictions(
        image=visualized_image,
        prediction=prediction,
        class_names=class_names,
        score_threshold=score_threshold,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), visualized_image)

    return True


def visualize_from_split(
    model,
    config: dict,
    data_root: Path,
    output_dir: Path,
    split_name: str,
    max_sequences: int | None,
    max_frames: int | None,
    score_threshold: float,
    device: torch.device,
) -> int:
    splits_config = config["splits"]

    split_file = Path(splits_config[split_name])
    if not split_file.is_absolute():
        split_file = resolve_project_path(split_file)

    sequence_paths = read_split_file(split_file)

    if max_sequences is not None:
        sequence_paths = sequence_paths[:max_sequences]

    class_names = config["data"]["class_names"]

    saved_count = 0

    for sequence_index, sequence_path in enumerate(sequence_paths, start=1):
        sequence_dir = resolve_sequence_path(
            data_root=data_root,
            sequence_path=sequence_path,
        )

        image_dir = sequence_dir / "img1"

        if not image_dir.exists():
            print(f"[WARNING] Missing img1 directory: {image_dir}")
            continue

        image_paths = list_images(image_dir)

        if max_frames is not None:
            image_paths = image_paths[:max_frames]

        print(f"[{sequence_index}/{len(sequence_paths)}] {sequence_dir.name}: {len(image_paths)} frames")

        for image_path in image_paths:
            output_path = output_dir / sequence_dir.name / image_path.name

            saved = save_visualized_image(
                model=model,
                image_path=image_path,
                output_path=output_path,
                class_names=class_names,
                score_threshold=score_threshold,
                device=device,
            )

            if saved:
                print(f"Saved: {output_path}")
                saved_count += 1

    return saved_count


def visualize_from_source(
    model,
    config: dict,
    source_path: Path,
    output_dir: Path,
    max_frames: int | None,
    score_threshold: float,
    device: torch.device,
) -> int:
    class_names = config["data"]["class_names"]
    image_paths = collect_source_images(source_path, max_frames=max_frames)

    saved_count = 0

    for image_path in image_paths:
        if source_path.is_file():
            relative_output_path = Path(image_path.name)
        else:
            try:
                relative_output_path = image_path.relative_to(source_path)
            except ValueError:
                relative_output_path = Path(image_path.name)

        output_path = output_dir / relative_output_path

        saved = save_visualized_image(
            model=model,
            image_path=image_path,
            output_path=output_path,
            class_names=class_names,
            score_threshold=score_threshold,
            device=device,
        )

        if saved:
            print(f"Saved: {output_path}")
            saved_count += 1

    return saved_count


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    experiment_name = config["experiment_name"]

    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = resolve_project_path(checkpoint_path)

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = resolve_project_path(data_root)

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = resolve_project_path(output_root)

    output_dir = output_root / experiment_name / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    num_classes = int(config["model"]["num_classes"])

    model = create_faster_rcnn_model(num_classes=num_classes)
    model = load_checkpoint(model, checkpoint_path, device=device)
    model.to(device)
    model.eval()

    print("=== FASTER R-CNN VISUALIZATION START ===")
    print(f"experiment      : {experiment_name}")
    print(f"checkpoint      : {checkpoint_path}")
    print(f"data root       : {data_root}")
    print(f"output dir      : {output_dir}")
    print(f"device          : {device}")
    print(f"score threshold : {args.score_threshold}")
    print("========================================")

    if args.source is not None:
        source_path = Path(args.source)
        if not source_path.is_absolute():
            source_path = resolve_project_path(source_path)

        saved_count = visualize_from_source(
            model=model,
            config=config,
            source_path=source_path,
            output_dir=output_dir,
            max_frames=args.max_frames,
            score_threshold=args.score_threshold,
            device=device,
        )
    else:
        saved_count = visualize_from_split(
            model=model,
            config=config,
            data_root=data_root,
            output_dir=output_dir,
            split_name=args.split,
            max_sequences=args.max_sequences,
            max_frames=args.max_frames,
            score_threshold=args.score_threshold,
            device=device,
        )

    print()
    print(f"Visualization finished. Saved images: {saved_count}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()