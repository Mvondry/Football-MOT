from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image


class FootballDataset(torch.utils.data.Dataset):
    """
    Dataset class for SoccerNet-Tracking sequences stored in MOT-like format.

    The split file is expected to contain paths to SNMOT sequence folders.
    Each sequence folder must contain:
        - img1/
        - gt/<gt_filename>

    Bounding boxes are loaded from MOT format (x, y, w, h) and converted to
    TorchVision Faster R-CNN format (x1, y1, x2, y2).
    """

    def __init__(
        self,
        root_dir,
        split_file,
        gt_filename,
        transforms=None,
        class_column=10,
        filter_invalid_boxes=True,
    ):
        self.root_dir = Path(root_dir)
        self.split_file = Path(split_file)
        self.gt_filename = gt_filename
        self.transforms = transforms
        self.class_column = class_column
        self.filter_invalid_boxes = filter_invalid_boxes

        self.imgs_info = []
        self._load_dataset()

    def _resolve_sequence_path(self, sequence_path):
        sequence_path = Path(sequence_path)

        if sequence_path.is_absolute():
            return sequence_path

        return self.root_dir / sequence_path

    def _load_dataset(self):
        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")

        with open(self.split_file, "r", encoding="utf-8") as f:
            sequence_paths = [line.strip() for line in f if line.strip()]

        for sequence_path in sequence_paths:
            sequence_dir = self._resolve_sequence_path(sequence_path)

            gt_path = sequence_dir / "gt" / self.gt_filename
            img_dir = sequence_dir / "img1"

            if not gt_path.exists():
                continue

            try:
                df = pd.read_csv(gt_path, header=None)
            except pd.errors.EmptyDataError:
                continue

            frame_ids = df[0].unique()

            for frame_id in frame_ids:
                img_name = f"{int(frame_id):06d}.jpg"
                img_path = img_dir / img_name

                if not img_path.exists():
                    continue

                frame_data = df[df[0] == frame_id].values

                self.imgs_info.append(
                    {
                        "img_path": img_path,
                        "boxes": frame_data[:, 2:6],
                        "labels": frame_data[:, self.class_column],
                        "image_id": len(self.imgs_info),
                    }
                )

    def __len__(self):
        return len(self.imgs_info)

    def __getitem__(self, idx):
        info = self.imgs_info[idx]

        image = Image.open(info["img_path"]).convert("RGB")

        boxes = info["boxes"].astype(np.float32)
        labels = info["labels"].astype(np.int64)

        # MOT format: (x, y, w, h)
        # TorchVision Faster R-CNN format: (x1, y1, x2, y2)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "area": torch.as_tensor(
                (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
                dtype=torch.float32,
            ),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        if self.filter_invalid_boxes:
            target = self._filter_invalid_boxes(target)

        return image, target

    @staticmethod
    def _filter_invalid_boxes(target):
        boxes = target["boxes"]

        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])

        target["boxes"] = target["boxes"][keep]
        target["labels"] = target["labels"][keep]
        target["area"] = target["area"][keep]
        target["iscrowd"] = target["iscrowd"][keep]

        return target

    def get_height_and_width(self, idx):
        image_path = self.imgs_info[idx]["img_path"]
        image = Image.open(image_path)
        return image.height, image.width