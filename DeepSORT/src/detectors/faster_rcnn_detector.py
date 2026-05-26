from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FasterRCNNDetector:
    """
    Wrapper for Faster R-CNN detector used before DeepSORT.

    Returns detections in xyxy format:
        det_xyxy:    ndarray (N, 4)
        det_confs:   ndarray (N,)
        det_classes: ndarray (N,)

    Faster R-CNN class labels are one-based:
        0 = background
        1..K = object classes
    """

    def __init__(
        self,
        weights_path: str | Path,
        num_classes: int,
        confidence_threshold: float = 0.35,
        device: str | torch.device = "cpu",
        keep_labels: list[int] | None = None,
    ):
        self.weights_path = Path(weights_path)
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.device = torch.device(device)
        self.keep_labels = set(keep_labels) if keep_labels is not None else None

        if not self.weights_path.exists():
            raise FileNotFoundError(f"Faster R-CNN weights not found: {self.weights_path}")

        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

    def _create_model(self):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features,
            self.num_classes,
        )

        return model

    def _load_model(self):
        model = self._create_model()

        checkpoint = torch.load(
            str(self.weights_path),
            map_location=self.device,
        )

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        return model

    def detect(self, frame_bgr):
        frame_height, frame_width = frame_bgr.shape[:2]

        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb).to(self.device)
        image_tensor = image_tensor.permute(2, 0, 1).float() / 255.0

        with torch.no_grad():
            prediction = self.model([image_tensor])[0]

        boxes = prediction.get("boxes", None)
        scores = prediction.get("scores", None)
        labels = prediction.get("labels", None)

        if boxes is None or scores is None or labels is None:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )

        boxes = boxes.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy().astype(np.int32)

        det_xyxy = []
        det_confs = []
        det_classes = []

        for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels):
            score = float(score)
            label = int(label)

            if score < self.confidence_threshold:
                continue

            if label == 0:
                continue

            if self.keep_labels is not None and label not in self.keep_labels:
                continue

            x1 = float(max(0, min(x1, frame_width - 1)))
            x2 = float(max(0, min(x2, frame_width - 1)))
            y1 = float(max(0, min(y1, frame_height - 1)))
            y2 = float(max(0, min(y2, frame_height - 1)))

            if x2 <= x1 or y2 <= y1:
                continue

            det_xyxy.append([x1, y1, x2, y2])
            det_confs.append(score)
            det_classes.append(label)

        return (
            np.array(det_xyxy, dtype=np.float32),
            np.array(det_confs, dtype=np.float32),
            np.array(det_classes, dtype=np.int32),
        )