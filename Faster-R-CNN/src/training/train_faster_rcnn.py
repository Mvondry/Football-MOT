import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

TORCHVISION_REFERENCES = PROJECT_ROOT / "external" / "torchvision_references"
sys.path.insert(0, str(TORCHVISION_REFERENCES))

from src.datasets.football_dataset import FootballDataset
from src.models.faster_rcnn import create_faster_rcnn_model

try:
    from my_engine import train_one_epoch, evaluate
except ImportError:
    from engine import train_one_epoch, evaluate

import utils
import transforms as T


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Faster R-CNN on SoccerNet-Tracking data."
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
        required=True,
        help="Root directory containing SoccerNet-Tracking sequence folders.",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="results",
        help="Root directory for experiment outputs.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use, e.g. 'cuda' or 'cpu'. If not set, it is chosen automatically.",
    )

    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable logging to Weights & Biases.",
    )

    parser.add_argument(
        "--save-all-checkpoints",
        action="store_true",
        help="Save checkpoint after every epoch.",
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


def get_transform(train, augmentation_config):
    transforms = [
        T.PILToTensor(),
        T.ToDtype(torch.float, scale=True),
    ]

    if train and augmentation_config.get("enabled", False):
        zoom_p = augmentation_config.get("random_zoom_out", 0.0)
        photometric_p = augmentation_config.get("random_photometric_distort", 0.0)
        flip_p = augmentation_config.get("random_horizontal_flip", 0.0)

        if zoom_p > 0:
            transforms.append(T.RandomZoomOut(side_range=(1.0, 2.0), p=zoom_p))

        if photometric_p > 0:
            transforms.append(T.RandomPhotometricDistort(p=photometric_p))

        if flip_p > 0:
            transforms.append(T.RandomHorizontalFlip(p=flip_p))

    return T.Compose(transforms)


def prepare_output_dirs(output_root, experiment_name):
    output_dir = Path(output_root) / experiment_name
    checkpoints_dir = output_dir / "checkpoints"
    logs_dir = output_dir / "logs"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    return output_dir, checkpoints_dir, logs_dir


def save_checkpoint(path, model, optimizer, lr_scheduler, epoch, metrics, config):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "metrics": metrics,
        "config": config,
    }

    torch.save(checkpoint, path)


def init_wandb(config, args):
    if not args.use_wandb:
        return None

    import wandb

    run = wandb.init(
        project="faster-rcnn-football",
        name=config["experiment_name"],
        config=config,
    )

    return run


def main():
    args = parse_args()
    config = load_config(args.config)

    experiment_name = config["experiment_name"]
    output_dir, checkpoints_dir, logs_dir = prepare_output_dirs(
        args.output_root,
        experiment_name,
    )

    shutil.copy2(args.config, output_dir / "config_used.yaml")

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(f"Experiment: {experiment_name}")
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")

    wandb_run = init_wandb(config, args)

    train_split = resolve_project_path(config["splits"]["train"])
    val_split = resolve_project_path(config["splits"]["val"])

    gt_filename = config["data"]["gt_filename"]
    augmentation_config = config.get("augmentation", {})

    dataset_train = FootballDataset(
        root_dir=args.data_root,
        split_file=train_split,
        gt_filename=gt_filename,
        transforms=get_transform(train=True, augmentation_config=augmentation_config),
    )

    dataset_val = FootballDataset(
        root_dir=args.data_root,
        split_file=val_split,
        gt_filename=gt_filename,
        transforms=get_transform(train=False, augmentation_config=augmentation_config),
    )

    print(f"Training images: {len(dataset_train)}")
    print(f"Validation images: {len(dataset_val)}")

    if len(dataset_train) == 0:
        raise RuntimeError("Training dataset is empty. Check data root, split file and GT filename.")

    if len(dataset_val) == 0:
        raise RuntimeError("Validation dataset is empty. Check data root, split file and GT filename.")

    training_config = config["training"]

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=training_config["num_workers"],
        collate_fn=utils.collate_fn,
    )

    data_loader_val = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=training_config["num_workers"],
        collate_fn=utils.collate_fn,
    )

    model = create_faster_rcnn_model(
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"].get("pretrained", True),
    )

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params,
        lr=training_config["learning_rate"],
        momentum=training_config["momentum"],
        weight_decay=training_config["weight_decay"],
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=training_config["lr_step_size"],
        gamma=training_config["lr_gamma"],
    )

    csv_path = logs_dir / "training_log.csv"

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "mAP",
                "mAP_50",
                "mAP_75",
                "learning_rate",
                "time_sec",
            ]
        )

    best_map = -1.0
    best_checkpoint_path = checkpoints_dir / "best.pth"

    print("Starting training...")

    for epoch in range(training_config["epochs"]):
        start_time = time.time()

        metric_logger = train_one_epoch(
            model,
            optimizer,
            data_loader_train,
            device,
            epoch,
            print_freq=10,
        )

        lr_scheduler.step()

        coco_evaluator = evaluate(
            model,
            data_loader_val,
            device=device,
        )

        train_loss = metric_logger.meters["loss"].global_avg
        current_lr = optimizer.param_groups[0]["lr"]

        bbox_stats = coco_evaluator.coco_eval["bbox"].stats
        map_5095 = float(bbox_stats[0])
        map_50 = float(bbox_stats[1])
        map_75 = float(bbox_stats[2])

        epoch_time = time.time() - start_time

        metrics = {
            "train_loss": float(train_loss),
            "mAP": map_5095,
            "mAP_50": map_50,
            "mAP_75": map_75,
            "learning_rate": float(current_lr),
            "time_sec": float(epoch_time),
        }

        print(
            f"Epoch {epoch}: "
            f"loss={train_loss:.4f}, "
            f"mAP={map_5095:.4f}, "
            f"mAP50={map_50:.4f}, "
            f"mAP75={map_75:.4f}, "
            f"time={epoch_time:.1f}s"
        )

        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    train_loss,
                    map_5095,
                    map_50,
                    map_75,
                    current_lr,
                    epoch_time,
                ]
            )

        latest_checkpoint_path = checkpoints_dir / "last.pth"
        save_checkpoint(
            latest_checkpoint_path,
            model,
            optimizer,
            lr_scheduler,
            epoch,
            metrics,
            config,
        )

        if args.save_all_checkpoints:
            epoch_checkpoint_path = checkpoints_dir / f"epoch_{epoch}.pth"
            save_checkpoint(
                epoch_checkpoint_path,
                model,
                optimizer,
                lr_scheduler,
                epoch,
                metrics,
                config,
            )

        if map_5095 > best_map:
            best_map = map_5095
            save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                lr_scheduler,
                epoch,
                metrics,
                config,
            )
            print(f"New best checkpoint saved: {best_checkpoint_path}")

        if wandb_run is not None:
            import wandb

            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "mAP": map_5095,
                    "mAP_50": map_50,
                    "mAP_75": map_75,
                    "learning_rate": current_lr,
                    "epoch_time": epoch_time,
                }
            )

    if wandb_run is not None:
        wandb_run.finish()

    print("Training finished.")
    print(f"Best mAP: {best_map:.4f}")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Training log: {csv_path}")


if __name__ == "__main__":
    main()