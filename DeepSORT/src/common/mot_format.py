from __future__ import annotations

import numpy as np


def xyxy_to_xywh_center(bbox_xyxy) -> list[float]:
    x1, y1, x2, y2 = bbox_xyxy

    width = x2 - x1
    height = y2 - y1
    x_center = x1 + width / 2.0
    y_center = y1 + height / 2.0

    return [x_center, y_center, width, height]


def iou_xyxy(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    return inter_area / (area_a + area_b - inter_area + 1e-6)


def find_matching_detection(
    track_bbox_xyxy,
    detection_bboxes_xyxy,
    iou_threshold: float = 0.05,
) -> int:
    """
    Finds the detection with the highest IoU for a given tracker output box.

    Returns:
        Index of the matching detection or -1 if no detection is matched.
    """
    if len(detection_bboxes_xyxy) == 0:
        return -1

    ious = [
        iou_xyxy(track_bbox_xyxy, detection_bbox)
        for detection_bbox in detection_bboxes_xyxy
    ]

    best_index = int(np.argmax(ious))

    if ious[best_index] >= iou_threshold:
        return best_index

    return -1


def write_mot_line(
    file,
    frame_id: int,
    track_id: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    confidence: float = 1.0,
) -> None:
    """
    Writes one line in standard MOTChallenge format:

    frame, id, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1
    """
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)

    file.write(
        f"{frame_id},{track_id},"
        f"{x1:.2f},{y1:.2f},{width:.2f},{height:.2f},"
        f"{confidence:.6f},-1,-1,-1\n"
    )


def write_mot_line_with_class(
    file,
    frame_id: int,
    track_id: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    confidence: float = 1.0,
    class_id: int = -1,
) -> None:
    """
    Writes an extended MOT-like line for visualization:

    frame, id, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1, class_id

    This file should not be used directly for TrackEval.
    """
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)

    file.write(
        f"{frame_id},{track_id},"
        f"{x1:.2f},{y1:.2f},{width:.2f},{height:.2f},"
        f"{confidence:.6f},-1,-1,-1,{class_id}\n"
    )