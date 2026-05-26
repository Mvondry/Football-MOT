import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision
from torch.utils.data import DataLoader
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

TORCHVISION_REFERENCES = PROJECT_ROOT / "external" / "torchvision_references"
sys.path.insert(0, str(TORCHVISION_REFERENCES))

from src.datasets.football_dataset import FootballDataset
from src.models.faster_rcnn import create_faster_rcnn_model, load_faster_rcnn_checkpoint

try:
    from my_engine import evaluate
except ImportError:
    from engine import evaluate

import utils
import transforms as T


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Faster R-CNN on SoccerNet-Tracking data."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained Faster R-CNN checkpoint.",
    )

    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory containing SoccerNet-Tracking sequence folders.",
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
        help="Device to use, e.g. 'cuda' or 'cpu'. If not set, it is chosen automatically.",
    )

    return parser.parse_args()


def load_config(config_path):
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def resolve_project_path(path_value):
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def get_eval_transform():
    return T.Compose(
        [
            T.PILToTensor(),
            T.ToDtype(torch.float, scale=True),
        ]
    )


def prepare_output_dir(output_root, experiment_name):
    output_dir = Path(output_root) / experiment_name / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def match_detections_greedy(
    gt_boxes,
    pred_boxes,
    iou_threshold=0.5,
):
    """
    Greedy one-to-one matching by IoU.

    One ground-truth box can match at most one prediction and one prediction
    can match at most one ground-truth box.
    """
    matches = []

    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        unmatched_gt = list(range(len(gt_boxes)))
        unmatched_pred = list(range(len(pred_boxes)))
        return matches, unmatched_gt, unmatched_pred

    iou_matrix = torchvision.ops.box_iou(gt_boxes, pred_boxes)

    candidates = []

    for gt_idx in range(iou_matrix.shape[0]):
        for pred_idx in range(iou_matrix.shape[1]):
            iou_value = iou_matrix[gt_idx, pred_idx].item()

            if iou_value >= iou_threshold:
                candidates.append((iou_value, gt_idx, pred_idx))

    candidates.sort(key=lambda x: x[0], reverse=True)

    matched_gt = set()
    matched_pred = set()

    for iou_value, gt_idx, pred_idx in candidates:
        if gt_idx not in matched_gt and pred_idx not in matched_pred:
            matches.append((gt_idx, pred_idx, iou_value))
            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)

    unmatched_gt = [idx for idx in range(len(gt_boxes)) if idx not in matched_gt]
    unmatched_pred = [idx for idx in range(len(pred_boxes)) if idx not in matched_pred]

    return matches, unmatched_gt, unmatched_pred


def compute_detection_metrics_from_confusion_matrix(confusion_matrix, class_names):
    """
    Compute per-class precision, recall and F1 score from confusion matrix.

    Background class with index 0 is excluded from macro, micro and weighted averages.
    """
    per_class_results = []

    per_class_precisions = []
    per_class_recalls = []
    per_class_f1s = []
    per_class_supports = []

    for class_idx in range(1, len(class_names)):
        tp = int(confusion_matrix[class_idx, class_idx])
        fp = int(np.sum(confusion_matrix[:, class_idx]) - tp)
        fn = int(np.sum(confusion_matrix[class_idx, :]) - tp)
        support = int(tp + fn)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall > 0:
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = 0.0

        per_class_results.append(
            {
                "Class": class_names[class_idx],
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1_score,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "Support": support,
            }
        )

        per_class_precisions.append(precision)
        per_class_recalls.append(recall)
        per_class_f1s.append(f1_score)
        per_class_supports.append(support)

    tp_total = int(
        sum(confusion_matrix[class_idx, class_idx] for class_idx in range(1, len(class_names)))
    )

    fp_total = int(
        sum(
            np.sum(confusion_matrix[:, class_idx]) - confusion_matrix[class_idx, class_idx]
            for class_idx in range(1, len(class_names))
        )
    )

    fn_total = int(
        sum(
            np.sum(confusion_matrix[class_idx, :]) - confusion_matrix[class_idx, class_idx]
            for class_idx in range(1, len(class_names))
        )
    )

    support_total = int(sum(per_class_supports))

    precision_micro = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall_micro = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0

    if precision_micro + recall_micro > 0:
        f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro)
    else:
        f1_micro = 0.0

    precision_macro = float(np.mean(per_class_precisions)) if per_class_precisions else 0.0
    recall_macro = float(np.mean(per_class_recalls)) if per_class_recalls else 0.0
    f1_macro = float(np.mean(per_class_f1s)) if per_class_f1s else 0.0

    if support_total > 0:
        precision_weighted = float(np.average(per_class_precisions, weights=per_class_supports))
        recall_weighted = float(np.average(per_class_recalls, weights=per_class_supports))
        f1_weighted = float(np.average(per_class_f1s, weights=per_class_supports))
    else:
        precision_weighted = 0.0
        recall_weighted = 0.0
        f1_weighted = 0.0

    summary_results = [
        {
            "Class": "OVERALL_MICRO",
            "Precision": precision_micro,
            "Recall": recall_micro,
            "F1-Score": f1_micro,
            "TP": tp_total,
            "FP": fp_total,
            "FN": fn_total,
            "Support": support_total,
        },
        {
            "Class": "OVERALL_MACRO",
            "Precision": precision_macro,
            "Recall": recall_macro,
            "F1-Score": f1_macro,
            "TP": np.nan,
            "FP": np.nan,
            "FN": np.nan,
            "Support": support_total,
        },
        {
            "Class": "OVERALL_WEIGHTED",
            "Precision": precision_weighted,
            "Recall": recall_weighted,
            "F1-Score": f1_weighted,
            "TP": np.nan,
            "FP": np.nan,
            "FN": np.nan,
            "Support": support_total,
        },
    ]

    return per_class_results, summary_results


def build_confusion_matrix(
    model,
    data_loader,
    device,
    class_names,
    score_threshold,
    iou_threshold,
):
    model.eval()

    num_classes = len(class_names)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            images = [image.to(device) for image in images]
            outputs = model(images)

            for target, output in zip(targets, outputs):
                gt_boxes = target["boxes"].to(device)
                gt_labels = target["labels"].to(device)

                pred_boxes = output["boxes"].to(device)
                pred_labels = output["labels"].to(device)
                pred_scores = output["scores"].to(device)

                keep = pred_scores >= score_threshold
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]

                if len(gt_boxes) == 0 and len(pred_boxes) == 0:
                    continue

                if len(gt_boxes) == 0:
                    for pred_label in pred_labels:
                        pred_class = int(pred_label.item())
                        if 0 <= pred_class < num_classes:
                            confusion_matrix[0, pred_class] += 1
                    continue

                if len(pred_boxes) == 0:
                    for gt_label in gt_labels:
                        gt_class = int(gt_label.item())
                        if 0 <= gt_class < num_classes:
                            confusion_matrix[gt_class, 0] += 1
                    continue

                matches, unmatched_gt, unmatched_pred = match_detections_greedy(
                    gt_boxes=gt_boxes,
                    pred_boxes=pred_boxes,
                    iou_threshold=iou_threshold,
                )

                for gt_idx, pred_idx, _ in matches:
                    gt_class = int(gt_labels[gt_idx].item())
                    pred_class = int(pred_labels[pred_idx].item())

                    if 0 <= gt_class < num_classes and 0 <= pred_class < num_classes:
                        confusion_matrix[gt_class, pred_class] += 1

                for gt_idx in unmatched_gt:
                    gt_class = int(gt_labels[gt_idx].item())

                    if 0 <= gt_class < num_classes:
                        confusion_matrix[gt_class, 0] += 1

                for pred_idx in unmatched_pred:
                    pred_class = int(pred_labels[pred_idx].item())

                    if 0 <= pred_class < num_classes:
                        confusion_matrix[0, pred_class] += 1

            if batch_idx % 50 == 0:
                print(f"Processed batches: {batch_idx}")

    return confusion_matrix


def save_confusion_matrix(confusion_matrix, class_names, output_dir):
    csv_path = output_dir / "confusion_matrix.csv"
    png_path = output_dir / "confusion_matrix.png"

    df = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    df.to_csv(csv_path)

    plt.figure(figsize=(max(8, len(class_names) * 1.4), max(6, len(class_names) * 1.1)))
    sns.heatmap(
        df,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Prediction")
    plt.ylabel("Ground Truth")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    return csv_path, png_path


def save_global_metrics(coco_evaluator, output_dir, checkpoint_path, config):
    stats = coco_evaluator.coco_eval["bbox"].stats

    metrics = {
        "mAP_50_95": float(stats[0]),
        "mAP_50": float(stats[1]),
        "mAP_75": float(stats[2]),
        "mAP_small": float(stats[3]),
        "mAP_medium": float(stats[4]),
        "mAP_large": float(stats[5]),
    }

    txt_path = output_dir / "global_metrics.txt"
    csv_path = output_dir / "global_metrics.csv"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== GLOBAL COCO METRICS ===\n")
        f.write(f"Experiment: {config['experiment_name']}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n\n")

        for key, value in metrics.items():
            f.write(f"{key}: {value:.6f}\n")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(metrics.keys())
        writer.writerow(metrics.values())

    return metrics, txt_path, csv_path


def save_class_metrics(per_class_results, summary_results, output_dir):
    all_results = per_class_results + summary_results

    csv_path = output_dir / "per_class_and_overall_metrics.csv"
    txt_path = output_dir / "overall_metrics.txt"

    df = pd.DataFrame(all_results)
    df.to_csv(csv_path, index=False)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== OVERALL DETECTION METRICS FROM CONFUSION MATRIX ===\n\n")

        for row in summary_results:
            f.write(f"{row['Class']}\n")
            f.write(f"  Precision: {row['Precision']:.6f}\n")
            f.write(f"  Recall:    {row['Recall']:.6f}\n")
            f.write(f"  F1-Score:  {row['F1-Score']:.6f}\n")

            if not pd.isna(row["TP"]):
                f.write(f"  TP:        {int(row['TP'])}\n")
                f.write(f"  FP:        {int(row['FP'])}\n")
                f.write(f"  FN:        {int(row['FN'])}\n")

            f.write(f"  Support:   {int(row['Support'])}\n\n")

    return csv_path, txt_path


def main():
    args = parse_args()
    config = load_config(args.config)

    experiment_name = config["experiment_name"]
    output_dir = prepare_output_dir(args.output_root, experiment_name)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(f"Experiment: {experiment_name}")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {output_dir}")

    test_split = resolve_project_path(config["splits"]["test"])

    dataset_test = FootballDataset(
        root_dir=args.data_root,
        split_file=test_split,
        gt_filename=config["data"]["gt_filename"],
        transforms=get_eval_transform(),
    )

    if len(dataset_test) == 0:
        raise RuntimeError("Test dataset is empty. Check data root, split file and GT filename.")

    num_workers = config["training"].get("num_workers", 4)

    data_loader_test = DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=utils.collate_fn,
    )

    print(f"Test images: {len(dataset_test)}")

    model = create_faster_rcnn_model(
        num_classes=config["model"]["num_classes"],
        pretrained=False,
    )

    model = load_faster_rcnn_checkpoint(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    model.to(device)
    model.eval()

    print("Running COCO mAP evaluation...")
    coco_evaluator = evaluate(
        model,
        data_loader_test,
        device=device,
    )

    global_metrics, global_txt, global_csv = save_global_metrics(
        coco_evaluator=coco_evaluator,
        output_dir=output_dir,
        checkpoint_path=args.checkpoint,
        config=config,
    )

    print(f"Global metrics saved to: {global_txt}")
    print(f"Global metrics CSV saved to: {global_csv}")

    class_names = config["data"]["class_names"]

    score_threshold = config["evaluation"].get("score_threshold", 0.5)
    iou_threshold = config["evaluation"].get("iou_threshold", 0.5)

    print("Building confusion matrix...")
    confusion_matrix = build_confusion_matrix(
        model=model,
        data_loader=data_loader_test,
        device=device,
        class_names=class_names,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
    )

    cm_csv, cm_png = save_confusion_matrix(
        confusion_matrix=confusion_matrix,
        class_names=class_names,
        output_dir=output_dir,
    )

    print(f"Confusion matrix saved to: {cm_csv}")
    print(f"Confusion matrix image saved to: {cm_png}")

    per_class_results, summary_results = compute_detection_metrics_from_confusion_matrix(
        confusion_matrix=confusion_matrix,
        class_names=class_names,
    )

    metrics_csv, metrics_txt = save_class_metrics(
        per_class_results=per_class_results,
        summary_results=summary_results,
        output_dir=output_dir,
    )

    print(f"Class metrics saved to: {metrics_csv}")
    print(f"Overall metrics saved to: {metrics_txt}")

    print("Evaluation finished.")


if __name__ == "__main__":
    main()