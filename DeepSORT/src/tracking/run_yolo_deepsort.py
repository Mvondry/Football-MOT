from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.deepsort_utils import create_deepsort, normalize_deepsort_outputs
from src.common.io_utils import (
    list_images,
    read_split_file,
    read_seqinfo,
    resolve_sequence_path,
    write_seqmap,
)
from src.common.mot_format import (
    find_matching_detection,
    write_mot_line,
    write_mot_line_with_class,
    xyxy_to_xywh_center,
)
from src.detectors.yolo_detector import YOLODetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO detector with DeepSORT and export MOT-format tracking results."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
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
        "--device",
        type=str,
        default=None,
        help="Device for YOLO inference, e.g. '0' or 'cpu'.",
    )

    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Optional maximum number of sequences for debugging.",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames per sequence for debugging.",
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


def add_deepsort_repo_to_path(config: dict) -> None:
    external_config = config.get("external", {})
    deepsort_repo = external_config.get(
        "deep_sort_pytorch",
        PROJECT_ROOT / "external" / "deep_sort_pytorch",
    )

    deepsort_repo = resolve_project_path(deepsort_repo)

    if not deepsort_repo.exists():
        raise FileNotFoundError(
            f"deep_sort_pytorch repository not found: {deepsort_repo}\n"
            "Clone https://github.com/ZQPei/deep_sort_pytorch into external/deep_sort_pytorch "
            "or set external.deep_sort_pytorch in the config."
        )

    sys.path.insert(0, str(deepsort_repo))


def get_device(config: dict, args: argparse.Namespace) -> str:
    if args.device is not None:
        return args.device

    detector_config = config.get("detector", {})

    if "device" in detector_config:
        return str(detector_config["device"])

    return "0" if torch.cuda.is_available() else "cpu"


def prepare_output_dirs(output_root: Path, tracker_name: str, save_class_output: bool):
    tracker_data_dir = output_root / "trackers" / tracker_name / "data"
    tracker_data_dir.mkdir(parents=True, exist_ok=True)

    tracker_class_data_dir = None

    if save_class_output:
        tracker_class_data_dir = output_root / "trackers" / tracker_name / "data_with_classes"
        tracker_class_data_dir.mkdir(parents=True, exist_ok=True)

    timing_dir = output_root / "trackers" / tracker_name
    timing_dir.mkdir(parents=True, exist_ok=True)

    return tracker_data_dir, tracker_class_data_dir, timing_dir


def detections_to_deepsort_input(det_xyxy: np.ndarray) -> np.ndarray:
    if len(det_xyxy) == 0:
        return np.empty((0, 4), dtype=np.float32)

    return np.array(
        [xyxy_to_xywh_center(box) for box in det_xyxy],
        dtype=np.float32,
    )


def write_timing_summary(timing_path: Path, timing_rows: list[dict]) -> None:
    timing_path.parent.mkdir(parents=True, exist_ok=True)

    with open(timing_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sequence",
                "frames",
                "total_time_sec",
                "avg_time_per_frame_sec",
                "fps",
            ],
        )

        writer.writeheader()
        writer.writerows(timing_rows)


def run_sequence(
    sequence_dir: Path,
    detector: YOLODetector,
    deepsort_checkpoint: Path,
    deepsort_params: dict,
    tracker_data_dir: Path,
    tracker_class_data_dir: Path | None,
    class_match_iou_threshold: float,
    max_frames: int | None = None,
    debug: bool = False,
) -> dict:
    sequence_name = sequence_dir.name
    image_dir = sequence_dir / "img1"

    if not image_dir.exists():
        print(f"[WARNING] Missing img1 directory for {sequence_name}, skipping.")
        return {
            "sequence": sequence_name,
            "frames": 0,
            "total_time_sec": 0.0,
            "avg_time_per_frame_sec": 0.0,
            "fps": 0.0,
        }

    image_paths = list_images(image_dir)

    if max_frames is not None:
        image_paths = image_paths[:max_frames]

    if not image_paths:
        print(f"[WARNING] No images found in {image_dir}, skipping.")
        return {
            "sequence": sequence_name,
            "frames": 0,
            "total_time_sec": 0.0,
            "avg_time_per_frame_sec": 0.0,
            "fps": 0.0,
        }

    seqinfo = read_seqinfo(sequence_dir)

    print()
    print(f"=== Sequence: {sequence_name} ===")
    print(f"Frames: {len(image_paths)} | source_fps={seqinfo['frameRate']}")

    deepsort = create_deepsort(
        checkpoint_path=deepsort_checkpoint,
        params=deepsort_params,
        use_cuda=torch.cuda.is_available(),
    )

    output_path = tracker_data_dir / f"{sequence_name}.txt"

    class_output_file = None
    class_output_path = None

    if tracker_class_data_dir is not None:
        class_output_path = tracker_class_data_dir / f"{sequence_name}.txt"
        class_output_file = open(class_output_path, "w", encoding="utf-8", newline="")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    sequence_start = time.time()

    processed_frames = 0

    with open(output_path, "w", encoding="utf-8", newline="") as mot_file:
        try:
            for local_frame_index, image_path in enumerate(image_paths, start=1):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                frame_start = time.time()

                frame = cv2.imread(str(image_path))

                if frame is None:
                    print(f"[WARNING] Cannot read frame: {image_path}")
                    continue

                frame_id = int(image_path.stem) if image_path.stem.isdigit() else local_frame_index

                det_xyxy, det_confs, det_classes = detector.detect(frame)

                bbox_xywh = detections_to_deepsort_input(det_xyxy)
                class_dummy = np.zeros(len(bbox_xywh), dtype=np.int32)

                try:
                    outputs = deepsort.update(
                        bbox_xywh,
                        det_confs,
                        class_dummy,
                        frame,
                    )
                except Exception as exc:
                    print(
                        f"[ERROR] DeepSORT failed on {sequence_name} "
                        f"frame {frame_id}: {exc}"
                    )
                    outputs = None

                tracks_matrix = normalize_deepsort_outputs(
                    outputs,
                    debug=debug,
                )

                det_xyxy_list = det_xyxy.tolist() if len(det_xyxy) else []

                for track in tracks_matrix:
                    x1, y1, x2, y2 = track[0:4]
                    track_id = int(track[5])

                    match_index = find_matching_detection(
                        track_bbox_xyxy=[x1, y1, x2, y2],
                        detection_bboxes_xyxy=det_xyxy_list,
                        iou_threshold=class_match_iou_threshold,
                    )

                    if match_index != -1:
                        confidence = float(det_confs[match_index])
                        class_id = int(det_classes[match_index])
                    else:
                        confidence = 1.0
                        class_id = -1

                    write_mot_line(
                        file=mot_file,
                        frame_id=frame_id,
                        track_id=track_id,
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        confidence=confidence,
                    )

                    if class_output_file is not None:
                        write_mot_line_with_class(
                            file=class_output_file,
                            frame_id=frame_id,
                            track_id=track_id,
                            x1=float(x1),
                            y1=float(y1),
                            x2=float(x2),
                            y2=float(y2),
                            confidence=confidence,
                            class_id=class_id,
                        )

                processed_frames += 1

                if debug and (processed_frames <= 3 or processed_frames % 50 == 0):
                    frame_time = time.time() - frame_start
                    print(
                        f"[{sequence_name}] frame={frame_id} "
                        f"dets={len(det_xyxy)} tracks={len(tracks_matrix)} "
                        f"time={frame_time:.3f}s"
                    )

        finally:
            if class_output_file is not None:
                class_output_file.close()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    sequence_time = time.time() - sequence_start
    avg_time = sequence_time / processed_frames if processed_frames > 0 else 0.0
    fps = processed_frames / sequence_time if sequence_time > 0 else 0.0

    print(f"Saved MOT output: {output_path}")

    if class_output_path is not None:
        print(f"Saved class output: {class_output_path}")

    print(
        f"Sequence time: {sequence_time:.2f}s | "
        f"avg/frame: {avg_time:.4f}s | FPS: {fps:.2f}"
    )

    return {
        "sequence": sequence_name,
        "frames": processed_frames,
        "total_time_sec": sequence_time,
        "avg_time_per_frame_sec": avg_time,
        "fps": fps,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    add_deepsort_repo_to_path(config)

    experiment_name = config["experiment_name"]

    data_config = config["data"]
    detector_config = config["detector"]
    deepsort_config = config["deepsort"]
    output_config = config.get("output", {})
    trackeval_config = config.get("trackeval", {})

    data_root = Path(args.data_root) if args.data_root is not None else Path(data_config["data_root"])
    split_file = Path(args.split_file) if args.split_file is not None else Path(data_config["split_file"])

    if not data_root.is_absolute():
        data_root = resolve_project_path(data_root)

    if not split_file.is_absolute():
        split_file = resolve_project_path(split_file)

    output_root = (
        Path(args.output_root)
        if args.output_root is not None
        else Path(trackeval_config.get("output_root", "trackeval_data"))
    )

    if not output_root.is_absolute():
        output_root = resolve_project_path(output_root)

    tracker_name = trackeval_config.get("tracker_name", experiment_name)

    seqmap_name = trackeval_config.get("seqmap_name", f"{split_file.stem}_seqmap.txt")
    if not seqmap_name.endswith(".txt"):
        seqmap_name = f"{seqmap_name}.txt"

    seqmap_path = output_root / "seqmaps" / seqmap_name

    save_class_output = bool(output_config.get("save_class_output", True))
    class_match_iou_threshold = float(output_config.get("class_match_iou_threshold", 0.05))
    debug = bool(output_config.get("debug", False))

    tracker_data_dir, tracker_class_data_dir, timing_dir = prepare_output_dirs(
        output_root=output_root,
        tracker_name=tracker_name,
        save_class_output=save_class_output,
    )

    device = get_device(config, args)

    yolo_weights = Path(detector_config["weights"])
    if not yolo_weights.is_absolute():
        yolo_weights = resolve_project_path(yolo_weights)

    deepsort_checkpoint = Path(deepsort_config["checkpoint"])
    if not deepsort_checkpoint.is_absolute():
        deepsort_checkpoint = resolve_project_path(deepsort_checkpoint)

    keep_classes = detector_config.get("keep_classes", None)

    detector = YOLODetector(
        weights_path=yolo_weights,
        image_size=int(detector_config.get("image_size", 640)),
        confidence_threshold=float(detector_config.get("confidence_threshold", 0.35)),
        device=device,
        keep_classes=keep_classes,
    )

    sequence_paths = read_split_file(split_file)

    if args.max_sequences is not None:
        sequence_paths = sequence_paths[: args.max_sequences]

    sequence_dirs = [
        resolve_sequence_path(data_root, sequence_path)
        for sequence_path in sequence_paths
    ]

    sequence_names = [sequence_dir.name for sequence_dir in sequence_dirs]
    write_seqmap(sequence_names, seqmap_path)

    print("=== YOLO + DeepSORT tracking ===")
    print(f"Experiment:          {experiment_name}")
    print(f"Tracker name:        {tracker_name}")
    print(f"Data root:           {data_root}")
    print(f"Split file:          {split_file}")
    print(f"YOLO weights:        {yolo_weights}")
    print(f"DeepSORT checkpoint: {deepsort_checkpoint}")
    print(f"Device:              {device}")
    print(f"Output root:         {output_root}")
    print(f"Tracker data dir:    {tracker_data_dir}")
    print(f"Seqmap file:         {seqmap_path}")
    print()

    timing_rows = []

    for sequence_index, sequence_dir in enumerate(sequence_dirs, start=1):
        print(f"[{sequence_index}/{len(sequence_dirs)}] Processing {sequence_dir.name}")

        timing_row = run_sequence(
            sequence_dir=sequence_dir,
            detector=detector,
            deepsort_checkpoint=deepsort_checkpoint,
            deepsort_params=deepsort_config.get("params", {}),
            tracker_data_dir=tracker_data_dir,
            tracker_class_data_dir=tracker_class_data_dir,
            class_match_iou_threshold=class_match_iou_threshold,
            max_frames=args.max_frames,
            debug=debug,
        )

        timing_rows.append(timing_row)

    timing_path = timing_dir / "timing_summary.csv"
    write_timing_summary(timing_path, timing_rows)

    print()
    print("Done.")
    print(f"Timing summary: {timing_path}")
    print()
    print("Use these values for HOTA evaluation:")
    print(f"  TRACKERS_FOLDER    = {output_root / 'trackers'}")
    print(f"  TRACKERS_TO_EVAL   = ['{tracker_name}']")
    print("  TRACKER_SUB_FOLDER = data")
    print(f"  SEQMAP_FILE        = {seqmap_path}")


if __name__ == "__main__":
    main()