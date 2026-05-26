from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO


class YOLODetector:
    """
    Wrapper for YOLO detector used before DeepSORT.

    Returns detections in xyxy format:
        det_xyxy:    ndarray (N, 4)
        det_confs:   ndarray (N,)
        det_classes: ndarray (N,)

    YOLO class IDs are zero-based:
        0 = player
        1 = goalkeeper
        2 = ball
        3 = referee
    """

    def __init__(
        self,
        weights_path: str | Path,
        image_size: int = 640,
        confidence_threshold: float = 0.35,
        device: str = "cpu",
        keep_classes: list[int] | None = None,
    ):
        self.weights_path = Path(weights_path)
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.keep_classes = set(keep_classes) if keep_classes is not None else None

        if not self.weights_path.exists():
            raise FileNotFoundError(f"YOLO weights not found: {self.weights_path}")

        self.model = YOLO(str(self.weights_path))

    def detect(self, frame_bgr):
        frame_height, frame_width = frame_bgr.shape[:2]

        prediction = self.model.predict(
            source=frame_bgr,
            imgsz=self.image_size,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )[0]

        det_xyxy = []
        det_confs = []
        det_classes = []

        if prediction.boxes is None or len(prediction.boxes) == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )

        boxes_xyxy = prediction.boxes.xyxy.detach().cpu().numpy()
        confidences = prediction.boxes.conf.detach().cpu().numpy()
        classes = prediction.boxes.cls.detach().cpu().numpy().astype(int)

        for (x1, y1, x2, y2), confidence, class_id in zip(
            boxes_xyxy,
            confidences,
            classes,
        ):
            class_id = int(class_id)
            confidence = float(confidence)

            if self.keep_classes is not None and class_id not in self.keep_classes:
                continue

            x1 = float(max(0, min(x1, frame_width - 1)))
            x2 = float(max(0, min(x2, frame_width - 1)))
            y1 = float(max(0, min(y1, frame_height - 1)))
            y2 = float(max(0, min(y2, frame_height - 1)))

            if x2 <= x1 or y2 <= y1:
                continue

            det_xyxy.append([x1, y1, x2, y2])
            det_confs.append(confidence)
            det_classes.append(class_id)

        return (
            np.array(det_xyxy, dtype=np.float32),
            np.array(det_confs, dtype=np.float32),
            np.array(det_classes, dtype=np.int32),
        )