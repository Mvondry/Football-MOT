from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import ultralytics
import yaml
from ultralytics import YOLO, settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO model on SoccerNet-Tracking YOLO dataset."
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
        help="Path to trained YOLO weights, for example results/YOLO26s-aug/weights/best.pt.",
    )

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Optional path to dataset YAML. Overrides value from config.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split used for evaluation.",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="results",
        help="Root directory for evaluation outputs.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use, e.g. '0' or 'cpu'. Overrides value from config.",
    )

    parser.add_argument(
        "--plots",
        action="store_true",
        help="Save Ultralytics plots such as confusion matrix and PR curves.",
    )

    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save COCO-style JSON if supported by Ultralytics.",
    )

    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow overwriting an existing evaluation directory.",
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


def as_float(value) -> float:
    if hasattr(value, "item"):
        return float(value.item())

    return float(value)


def get_class_names(metrics, model) -> dict[int, str]:
    names = getattr(metrics, "names", None)

    if names is None:
        names = getattr(model, "names", {})

    return {int(class_id): str(class_name) for class_id, class_name in dict(names).items()}


def build_per_class_dataframe(metrics, model) -> pd.DataFrame:
    """
    Build a stable per-class table:
    class_id, class, instances, precision, recall, mAP50, mAP50-95.
    """
    names = get_class_names(metrics, model)

    nt_per_class = getattr(metrics, "nt_per_class", None)
    if nt_per_class is not None:
        nt_per_class = np.asarray(nt_per_class)

    ap_class_index = getattr(metrics, "ap_class_index", None)

    if ap_class_index is None:
        ap_class_index = getattr(metrics.box, "ap_class_index", None)

    if ap_class_index is None:
        ap_class_index = list(range(len(names)))

    rows = []

    for metric_index, class_id in enumerate(ap_class_index):
        class_id = int(class_id)
        class_name = names.get(class_id, str(class_id))

        try:
            precision, recall, map50, map5095 = metrics.class_result(metric_index)
        except Exception:
            precision, recall, map50, map5095 = metrics.box.class_result(metric_index)

        if nt_per_class is not None and class_id < len(nt_per_class):
            instances = int(nt_per_class[class_id])
        else:
            instances = None

        rows.append(
            {
                "class_id": class_id,
                "class": class_name,
                "instances": instances,
                "precision": as_float(precision),
                "recall": as_float(recall),
                "mAP50": as_float(map50),
                "mAP50-95": as_float(map5095),
            }
        )

    try:
        precision, recall, map50, map5095 = metrics.mean_results()
    except Exception:
        precision, recall, map50, map5095 = metrics.box.mean_results()

    total_instances = int(np.sum(nt_per_class)) if nt_per_class is not None else None

    rows.append(
        {
            "class_id": -1,
            "class": "all",
            "instances": total_instances,
            "precision": as_float(precision),
            "recall": as_float(recall),
            "mAP50": as_float(map50),
            "mAP50-95": as_float(map5095),
        }
    )

    return pd.DataFrame(rows)


def save_text_summary(
    path: Path,
    summary: dict,
    per_class_df: pd.DataFrame,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== YOLO EVALUATION SUMMARY ===\n\n")

        f.write("Configuration\n")
        f.write(f"  experiment: {summary['experiment_name']}\n")
        f.write(f"  weights:    {summary['weights']}\n")
        f.write(f"  data:       {summary['data']}\n")
        f.write(f"  split:      {summary['split']}\n")
        f.write(f"  imgsz:      {summary['imgsz']}\n")
        f.write(f"  batch:      {summary['batch']}\n")
        f.write(f"  device:     {summary['device']}\n\n")

        f.write("Global metrics\n")
        f.write(f"  precision: {summary['precision']:.6f}\n")
        f.write(f"  recall:    {summary['recall']:.6f}\n")
        f.write(f"  mAP50:     {summary['mAP50']:.6f}\n")
        f.write(f"  mAP50-95:  {summary['mAP50-95']:.6f}\n")
        f.write(f"  mAP75:     {summary['mAP75']:.6f}\n\n")

        f.write("Speed [ms/image]\n")
        for key, value in summary["speed"].items():
            f.write(f"  {key}: {value:.6f}\n")

        f.write("\nPer-class metrics\n")
        f.write(per_class_df.to_string(index=False))
        f.write("\n")


def save_outputs(
    output_dir: Path,
    summary: dict,
    per_class_df: pd.DataFrame,
    metrics,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_json = output_dir / "metrics_summary.json"
    summary_txt = output_dir / "metrics_summary.txt"
    per_class_csv = output_dir / "per_class_metrics.csv"
    per_class_json = output_dir / "per_class_metrics.json"

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    save_text_summary(
        path=summary_txt,
        summary=summary,
        per_class_df=per_class_df,
    )

    per_class_df.to_csv(per_class_csv, index=False)
    per_class_df.to_json(
        per_class_json,
        orient="records",
        indent=4,
        force_ascii=False,
    )

    try:
        (output_dir / "ultralytics_summary.csv").write_text(
            metrics.to_csv(),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"[WARNING] Could not save metrics.to_csv(): {exc}")

    try:
        (output_dir / "ultralytics_summary.json").write_text(
            metrics.to_json(),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"[WARNING] Could not save metrics.to_json(): {exc}")

    print(f"Summary TXT:      {summary_txt}")
    print(f"Summary JSON:     {summary_json}")
    print(f"Per-class CSV:    {per_class_csv}")
    print(f"Per-class JSON:   {per_class_json}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    experiment_name = config["experiment_name"]

    output_root = resolve_project_path(args.output_root)
    output_dir = output_root / experiment_name / "evaluation" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = Path(args.data) if args.data is not None else Path(config["data"])
    if not data_yaml.is_absolute():
        data_yaml = resolve_project_path(data_yaml)

    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = resolve_project_path(weights_path)

    eval_config = config.get("evaluation", {})
    training_config = config.get("training", {})

    imgsz = eval_config.get("imgsz", training_config.get("imgsz", 640))
    batch = eval_config.get("batch", training_config.get("batch", 16))
    workers = eval_config.get("workers", training_config.get("workers", 2))
    device = args.device if args.device is not None else eval_config.get(
        "device",
        training_config.get("device", "0"),
    )

    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML does not exist: {data_yaml}")

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file does not exist: {weights_path}")

    settings.update(
        {
            "runs_dir": str(output_root),
            "wandb": False,
        }
    )

    print("=== YOLO EVALUATION START ===")
    print(f"ultralytics: {ultralytics.__version__}")
    print(f"torch      : {torch.__version__}")
    print(f"cuda avail : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"gpu        : {torch.cuda.get_device_name(0)}")

    print()
    print(f"experiment : {experiment_name}")
    print(f"weights    : {weights_path}")
    print(f"data       : {data_yaml}")
    print(f"split      : {args.split}")
    print(f"output dir : {output_dir}")
    print(f"device     : {device}")
    print("=============================")

    model = YOLO(str(weights_path))

    metrics = model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        project=str(output_dir),
        name="ultralytics_outputs",
        plots=args.plots,
        save_json=args.save_json,
        exist_ok=args.exist_ok,
        verbose=True,
    )

    try:
        precision, recall, map50, map5095 = metrics.mean_results()
    except Exception:
        precision, recall, map50, map5095 = metrics.box.mean_results()

    summary = {
        "experiment_name": experiment_name,
        "weights": str(weights_path.resolve()),
        "data": str(data_yaml.resolve()),
        "split": args.split,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "save_dir": str(Path(metrics.save_dir).resolve()),
        "precision": as_float(precision),
        "recall": as_float(recall),
        "mAP50": as_float(map50),
        "mAP50-95": as_float(map5095),
        "mAP75": as_float(metrics.box.map75),
        "speed": {
            key: as_float(value)
            for key, value in getattr(metrics, "speed", {}).items()
        },
    }

    per_class_df = build_per_class_dataframe(
        metrics=metrics,
        model=model,
    )

    save_outputs(
        output_dir=output_dir,
        summary=summary,
        per_class_df=per_class_df,
        metrics=metrics,
    )

    print("=== YOLO EVALUATION END ===")


if __name__ == "__main__":
    main()