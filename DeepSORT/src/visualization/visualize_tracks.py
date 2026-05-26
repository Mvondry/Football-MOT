from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.io_utils import list_images, read_split_file, resolve_sequence_path


CLASS_COLORS_BGR = {
    "player": (255, 0, 0),        # blue
    "goalkeeper": (0, 0, 255),    # red
    "ball": (255, 255, 0),        # cyan
    "referee": (0, 255, 255),     # yellow
    "other": (128, 128, 128),     # gray
    "unknown": (255, 255, 255),   # white
    "object": (255, 255, 255),    # white
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize DeepSORT tracking outputs on SoccerNet-Tracking frames."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to tracking YAML config.",
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Optional dataset root override.",
    )

    parser.add_argument(
        "--split-file",
        type=str,
        default=None,
        help="Optional split file override.",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Optional TrackEval output root override.",
    )

    parser.add_argument(
        "--tracker-name",
        type=str,
        default=None,
        help="Optional tracker name override.",
    )

    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Maximum number of sequences to visualize.",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=50,
        help="Maximum number of frames per sequence to visualize.",
    )

    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save an MP4 video for each visualized sequence.",
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


def load_tracks(track_file: Path) -> dict[int, list[dict]]:
    """
    Loads MOT-like tracking output.

    Expected standard format:
        frame,id,x,y,w,h,conf,-1,-1,-1

    Extended visualization format:
        frame,id,x,y,w,h,conf,-1,-1,-1,class_id
    """
    if not track_file.exists():
        raise FileNotFoundError(f"Track file not found: {track_file}")

    tracks_by_frame: dict[int, list[dict]] = {}

    with open(track_file, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split(",")

            if len(parts) < 10:
                print(f"[WARNING] Skipping invalid line {line_number} in {track_file}: {line}")
                continue

            try:
                frame_id = int(float(parts[0]))
                track_id = int(float(parts[1]))
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                confidence = float(parts[6])
                class_id = int(float(parts[10])) if len(parts) >= 11 else -1
            except ValueError:
                print(f"[WARNING] Could not parse line {line_number} in {track_file}: {line}")
                continue

            tracks_by_frame.setdefault(frame_id, []).append(
                {
                    "track_id": track_id,
                    "x1": x,
                    "y1": y,
                    "x2": x + w,
                    "y2": y + h,
                    "confidence": confidence,
                    "class_id": class_id,
                }
            )

    return tracks_by_frame


def get_class_name(class_id: int, class_names: list[str], class_id_offset: int) -> str:
    if class_id < 0:
        return "unknown"

    index = class_id - class_id_offset

    if 0 <= index < len(class_names):
        return class_names[index]

    return "object"


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

    if "object" in class_name:
        return "object"

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


def draw_tracks(
    frame,
    tracks: list[dict],
    class_names: list[str],
    class_id_offset: int,
    draw_confidence: bool,
) -> None:
    for track in tracks:
        track_id = int(track["track_id"])
        class_id = int(track["class_id"])
        confidence = float(track["confidence"])

        class_name = get_class_name(
            class_id=class_id,
            class_names=class_names,
            class_id_offset=class_id_offset,
        )

        if draw_confidence:
            label = f"{class_name} ID {track_id} {confidence:.2f}"
        else:
            label = f"{class_name} ID {track_id}"

        draw_box_with_label(
            image=frame,
            x1=track["x1"],
            y1=track["y1"],
            x2=track["x2"],
            y2=track["y2"],
            label=label,
            class_name=class_name,
        )


def create_video_writer(video_path: Path, frame_width: int, frame_height: int, fps: float):
    video_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    return cv2.VideoWriter(
        str(video_path),
        fourcc,
        fps,
        (frame_width, frame_height),
    )


def visualize_sequence(
    sequence_dir: Path,
    track_file: Path,
    output_sequence_dir: Path,
    class_names: list[str],
    class_id_offset: int,
    draw_confidence: bool,
    max_frames: int | None,
    save_video: bool,
    video_fps: float,
) -> None:
    image_dir = sequence_dir / "img1"

    if not image_dir.exists():
        print(f"[WARNING] Missing img1 directory: {image_dir}")
        return

    image_paths = list_images(image_dir)

    if max_frames is not None:
        image_paths = image_paths[:max_frames]

    if not image_paths:
        print(f"[WARNING] No images found in {image_dir}")
        return

    tracks_by_frame = load_tracks(track_file)

    output_sequence_dir.mkdir(parents=True, exist_ok=True)

    video_writer = None

    if save_video:
        first_frame = cv2.imread(str(image_paths[0]))

        if first_frame is None:
            raise RuntimeError(f"Could not read first frame: {image_paths[0]}")

        frame_height, frame_width = first_frame.shape[:2]
        video_path = output_sequence_dir.parent / f"{sequence_dir.name}.mp4"

        video_writer = create_video_writer(
            video_path=video_path,
            frame_width=frame_width,
            frame_height=frame_height,
            fps=video_fps,
        )

    saved_frames = 0

    for local_index, image_path in enumerate(image_paths, start=1):
        frame = cv2.imread(str(image_path))

        if frame is None:
            print(f"[WARNING] Could not read frame: {image_path}")
            continue

        frame_id = int(image_path.stem) if image_path.stem.isdigit() else local_index
        tracks = tracks_by_frame.get(frame_id, [])

        draw_tracks(
            frame=frame,
            tracks=tracks,
            class_names=class_names,
            class_id_offset=class_id_offset,
            draw_confidence=draw_confidence,
        )

        output_frame_path = output_sequence_dir / image_path.name
        cv2.imwrite(str(output_frame_path), frame)

        if video_writer is not None:
            video_writer.write(frame)

        saved_frames += 1

    if video_writer is not None:
        video_writer.release()

    print(f"[OK] {sequence_dir.name}: saved {saved_frames} visualized frames to {output_sequence_dir}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    experiment_name = config["experiment_name"]

    data_config = config["data"]
    trackeval_config = config.get("trackeval", {})
    visualization_config = config.get("visualization", {})

    data_root = Path(args.data_root) if args.data_root is not None else Path(data_config["data_root"])
    split_file = Path(args.split_file) if args.split_file is not None else Path(data_config["split_file"])

    if not data_root.is_absolute():
        data_root = resolve_project_path(data_root)

    if not split_file.is_absolute():
        split_file = resolve_project_path(split_file)

    output_root = (
        Path(args.output_root)
        if args.output_root is not None
        else Path(trackeval_config.get("output_root", "trackeval_data_results"))
    )

    if not output_root.is_absolute():
        output_root = resolve_project_path(output_root)

    tracker_name = (
        args.tracker_name
        if args.tracker_name is not None
        else trackeval_config.get("tracker_name", experiment_name)
    )

    track_subdir = visualization_config.get("track_subdir", "data_with_classes")

    class_names = visualization_config.get(
        "class_names",
        ["player", "goalkeeper", "ball", "referee"],
    )

    class_id_offset = int(visualization_config.get("class_id_offset", 0))
    draw_confidence = bool(visualization_config.get("draw_confidence", False))
    video_fps = float(visualization_config.get("video_fps", 25.0))

    tracker_output_dir = output_root / "trackers" / tracker_name / track_subdir
    visualization_output_dir = output_root / "trackers" / tracker_name / "visualizations"

    if not tracker_output_dir.exists():
        raise FileNotFoundError(
            f"Tracker output directory not found: {tracker_output_dir}\n"
            "Run tracking first and make sure data_with_classes output is enabled."
        )

    sequence_paths = read_split_file(split_file)

    if args.max_sequences is not None:
        sequence_paths = sequence_paths[: args.max_sequences]

    print("=== DeepSORT visualization ===")
    print(f"Experiment:        {experiment_name}")
    print(f"Tracker name:      {tracker_name}")
    print(f"Data root:         {data_root}")
    print(f"Split file:        {split_file}")
    print(f"Tracker outputs:   {tracker_output_dir}")
    print(f"Visualization dir: {visualization_output_dir}")
    print(f"Class offset:      {class_id_offset}")
    print("==============================")

    for sequence_index, sequence_path in enumerate(sequence_paths, start=1):
        sequence_dir = resolve_sequence_path(data_root, sequence_path)
        sequence_name = sequence_dir.name

        track_file = tracker_output_dir / f"{sequence_name}.txt"

        if not track_file.exists():
            print(f"[WARNING] Missing track file for {sequence_name}: {track_file}")
            continue

        output_sequence_dir = visualization_output_dir / sequence_name

        print(f"[{sequence_index}/{len(sequence_paths)}] Visualizing {sequence_name}")

        visualize_sequence(
            sequence_dir=sequence_dir,
            track_file=track_file,
            output_sequence_dir=output_sequence_dir,
            class_names=class_names,
            class_id_offset=class_id_offset,
            draw_confidence=draw_confidence,
            max_frames=args.max_frames,
            save_video=args.save_video,
            video_fps=video_fps,
        )

    print()
    print("Visualization finished.")
    print(f"Results saved to: {visualization_output_dir}")


if __name__ == "__main__":
    main()