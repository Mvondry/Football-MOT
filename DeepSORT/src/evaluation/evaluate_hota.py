from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DeepSORT tracking outputs using HOTA metric from sn-trackeval."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML tracking config.",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Optional TrackEval data root override.",
    )

    parser.add_argument(
        "--tracker-name",
        type=str,
        default=None,
        help="Optional tracker name override.",
    )

    parser.add_argument(
        "--seqmap-file",
        type=str,
        default=None,
        help="Optional seqmap file override.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split name used by TrackEval, usually 'test'.",
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        default="SNMOT",
        help="Benchmark name used by TrackEval.",
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel evaluation.",
    )

    parser.add_argument(
        "--num-cores",
        type=int,
        default=8,
        help="Number of cores used when --parallel is enabled.",
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


def add_sn_trackeval_to_path(config: dict) -> None:
    external_config = config.get("external", {})
    sn_trackeval_path = external_config.get(
        "sn_trackeval",
        PROJECT_ROOT / "external" / "sn-trackeval",
    )

    sn_trackeval_path = resolve_project_path(sn_trackeval_path)

    if not sn_trackeval_path.exists():
        raise FileNotFoundError(
            f"sn-trackeval repository not found: {sn_trackeval_path}\n"
            "Clone https://github.com/SoccerNet/sn-trackeval.git into external/sn-trackeval "
            "or set external.sn_trackeval in the config."
        )

    sys.path.insert(0, str(sn_trackeval_path))


def convert_summary_txt_to_csv(summary_txt_path: Path) -> Path | None:
    if not summary_txt_path.exists():
        print(f"[WARNING] Summary file not found: {summary_txt_path}")
        return None

    summary_csv_path = summary_txt_path.with_suffix(".csv")

    with open(summary_txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    rows = [line.split() for line in lines]

    with open(summary_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    return summary_csv_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    add_sn_trackeval_to_path(config)

    import trackeval

    experiment_name = config["experiment_name"]
    trackeval_config = config.get("trackeval", {})

    output_root = (
        Path(args.output_root)
        if args.output_root is not None
        else Path(trackeval_config.get("output_root", "trackeval_data"))
    )

    if not output_root.is_absolute():
        output_root = resolve_project_path(output_root)

    tracker_name = (
        args.tracker_name
        if args.tracker_name is not None
        else trackeval_config.get("tracker_name", experiment_name)
    )

    if args.seqmap_file is not None:
        seqmap_file = Path(args.seqmap_file)
        if not seqmap_file.is_absolute():
            seqmap_file = resolve_project_path(seqmap_file)
    else:
        seqmap_name = trackeval_config.get("seqmap_name", "test_seqmap.txt")

        if not seqmap_name.endswith(".txt"):
            seqmap_name = f"{seqmap_name}.txt"

        seqmap_file = output_root / "seqmaps" / seqmap_name

    gt_folder = output_root / "gt"
    trackers_folder = output_root / "trackers"
    tracker_data_folder = trackers_folder / tracker_name / "data"

    if not gt_folder.exists():
        raise FileNotFoundError(
            f"GT folder not found: {gt_folder}\n"
            "Run src/preprocessing/prepare_trackeval_gt.py first."
        )

    if not trackers_folder.exists():
        raise FileNotFoundError(f"Trackers folder not found: {trackers_folder}")

    if not tracker_data_folder.exists():
        raise FileNotFoundError(
            f"Tracker data folder not found: {tracker_data_folder}\n"
            "Run run_yolo_deepsort.py or run_frcnn_deepsort.py first."
        )

    if not seqmap_file.exists():
        raise FileNotFoundError(f"Seqmap file not found: {seqmap_file}")

    eval_config = trackeval.Evaluator.get_default_eval_config()
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    metrics_config = {"METRICS": ["HOTA"]}

    eval_config["USE_PARALLEL"] = bool(args.parallel)
    eval_config["NUM_PARALLEL_CORES"] = int(args.num_cores)
    eval_config["BREAK_ON_ERROR"] = True
    eval_config["PRINT_RESULTS"] = True
    eval_config["PRINT_ONLY_COMBINED"] = False
    eval_config["PRINT_CONFIG"] = True
    eval_config["TIME_PROGRESS"] = True
    eval_config["OUTPUT_SUMMARY"] = True
    eval_config["OUTPUT_DETAILED"] = True
    eval_config["PLOT_CURVES"] = True

    dataset_config["GT_FOLDER"] = str(gt_folder)
    dataset_config["TRACKERS_FOLDER"] = str(trackers_folder)
    dataset_config["TRACKERS_TO_EVAL"] = [tracker_name]
    dataset_config["TRACKER_SUB_FOLDER"] = "data"
    dataset_config["CLASSES_TO_EVAL"] = ["pedestrian"]
    dataset_config["BENCHMARK"] = args.benchmark
    dataset_config["SPLIT_TO_EVAL"] = args.split
    dataset_config["INPUT_AS_ZIP"] = False
    dataset_config["DO_PREPROC"] = False
    dataset_config["SEQMAP_FILE"] = str(seqmap_file)
    dataset_config["SKIP_SPLIT_FOL"] = True

    print("=== HOTA evaluation ===")
    print(f"Experiment:       {experiment_name}")
    print(f"Tracker name:     {tracker_name}")
    print(f"GT folder:        {gt_folder}")
    print(f"Trackers folder:  {trackers_folder}")
    print(f"Tracker data:     {tracker_data_folder}")
    print(f"Seqmap file:      {seqmap_file}")
    print(f"Benchmark:        {args.benchmark}")
    print(f"Split:            {args.split}")
    print("=======================")

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [trackeval.metrics.HOTA(metrics_config)]

    evaluator.evaluate(dataset_list, metrics_list)

    summary_txt_path = trackers_folder / tracker_name / "pedestrian_summary.txt"
    summary_csv_path = convert_summary_txt_to_csv(summary_txt_path)

    print()
    print("Evaluation finished.")
    print(f"Summary TXT: {summary_txt_path}")

    if summary_csv_path is not None:
        print(f"Summary CSV: {summary_csv_path}")


if __name__ == "__main__":
    main()