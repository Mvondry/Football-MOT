from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics import YOLO, settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO model on SoccerNet-Tracking data."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML training config.",
    )

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Optional path to YOLO dataset YAML. Overrides value from config.",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="results",
        help="Root directory for training outputs.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use, e.g. '0', '0,1' or 'cpu'. Overrides value from config.",
    )

    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )

    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow overwriting an existing Ultralytics run directory.",
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


def build_albumentations_transforms(augmentation_config: dict):
    """
    Builds optional custom Albumentations transforms.

    These transforms correspond to the camera/compression augmentations used
    during YOLO training.
    """
    albu_config = augmentation_config.get("albumentations", {})

    if not albu_config:
        return None

    import albumentations as A

    transforms = []

    if albu_config.get("blur", 0.0) > 0:
        transforms.append(A.Blur(p=albu_config["blur"]))

    if albu_config.get("median_blur", 0.0) > 0:
        transforms.append(A.MedianBlur(p=albu_config["median_blur"]))

    if albu_config.get("to_gray", 0.0) > 0:
        transforms.append(A.ToGray(p=albu_config["to_gray"]))

    if albu_config.get("clahe", 0.0) > 0:
        transforms.append(A.CLAHE(p=albu_config["clahe"]))

    if albu_config.get("random_brightness_contrast", 0.0) > 0:
        transforms.append(
            A.RandomBrightnessContrast(
                p=albu_config["random_brightness_contrast"]
            )
        )

    if albu_config.get("random_gamma", 0.0) > 0:
        transforms.append(A.RandomGamma(p=albu_config["random_gamma"]))

    if albu_config.get("image_compression", 0.0) > 0:
        transforms.append(A.ImageCompression(p=albu_config["image_compression"]))

    return transforms if transforms else None


def build_augmentation_args(augmentation_config: dict) -> dict:
    """
    Builds augmentation arguments passed to Ultralytics model.train().

    If augmentation.enabled is false, no additional thesis-specific
    augmentation arguments are passed.
    """
    if not augmentation_config.get("enabled", False):
        return {}

    augmentation_args = {}

    for key in [
        "mixup",
        "cutmix",
        "degrees",
        "shear",
        "perspective",
        "multi_scale",
    ]:
        if key in augmentation_config:
            augmentation_args[key] = augmentation_config[key]

    custom_albu = build_albumentations_transforms(augmentation_config)

    if custom_albu is not None:
        augmentation_args["augmentations"] = custom_albu

    return augmentation_args


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    experiment_name = config["experiment_name"]
    training_config = config["training"]
    augmentation_config = config.get("augmentation", {})

    output_root = resolve_project_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    experiment_dir = output_root / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(args.config, experiment_dir / "config_used.yaml")

    data_yaml = Path(args.data) if args.data is not None else Path(config["data"])
    if not data_yaml.is_absolute():
        data_yaml = resolve_project_path(data_yaml)

    model_name = config["model"]

    device = args.device if args.device is not None else training_config.get("device", "0")
    exist_ok = args.exist_ok or training_config.get("exist_ok", False)

    settings.update(
        {
            "runs_dir": str(output_root),
            "wandb": bool(args.use_wandb),
        }
    )

    print("=== YOLO TRAINING START ===")
    print(f"ultralytics: {ultralytics.__version__}")
    print(f"torch      : {torch.__version__}")
    print(f"cuda avail : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"gpu        : {torch.cuda.get_device_name(0)}")

    print()
    print(f"experiment : {experiment_name}")
    print(f"model      : {model_name}")
    print(f"data       : {data_yaml}")
    print(f"output root: {output_root}")
    print(f"device     : {device}")
    print(f"wandb      : {bool(args.use_wandb)}")
    print("===========================")

    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML file does not exist: {data_yaml}")

    model = YOLO(model_name)

    train_args = {
        "data": str(data_yaml),
        "epochs": training_config.get("epochs", 100),
        "imgsz": training_config.get("imgsz", 640),
        "batch": training_config.get("batch", 16),
        "device": device,
        "workers": training_config.get("workers", 6),
        "project": str(output_root),
        "name": experiment_name,
        "seed": training_config.get("seed", 0),
        "patience": training_config.get("patience", 50),
        "exist_ok": exist_ok,
        "plots": training_config.get("plots", False),
    }

    train_args.update(build_augmentation_args(augmentation_config))

    # Optional raw Ultralytics train arguments from config.
    # This allows adding or overriding any Ultralytics parameter without
    # changing this script.
    train_args.update(config.get("train_args", {}))

    model.train(**train_args)

    print("=== YOLO TRAINING END ===")
    print(f"Outputs saved under: {output_root / experiment_name}")


if __name__ == "__main__":
    main()