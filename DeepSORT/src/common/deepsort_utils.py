from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


DEFAULT_DEEPSORT_PARAMS = {
    "max_dist": 0.1,
    "min_confidence": 0.0,
    "nms_max_overlap": 1.0,
    "max_iou_distance": 0.85,
    "max_age": 40,
    "n_init": 4,
    "nn_budget": 100,
}


def create_deepsort(
    checkpoint_path: str | Path,
    params: dict | None = None,
    use_cuda: bool | None = None,
):
    """
    Creates DeepSORT tracker from deep_sort_pytorch.

    The tracking scripts must make sure that the deep_sort_pytorch package
    is available on PYTHONPATH before calling this function.
    """
    from deep_sort import DeepSort

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"DeepSORT checkpoint not found: {checkpoint_path}")

    deepsort_params = DEFAULT_DEEPSORT_PARAMS.copy()

    if params is not None:
        deepsort_params.update(params)

    if use_cuda is None:
        use_cuda = torch.cuda.is_available()

    return DeepSort(
        str(checkpoint_path),
        max_dist=deepsort_params["max_dist"],
        min_confidence=deepsort_params["min_confidence"],
        nms_max_overlap=deepsort_params["nms_max_overlap"],
        max_iou_distance=deepsort_params["max_iou_distance"],
        max_age=deepsort_params["max_age"],
        n_init=deepsort_params["n_init"],
        nn_budget=deepsort_params["nn_budget"],
        use_cuda=use_cuda,
    )


def _asarray_quiet(value):
    try:
        return np.asarray(value)
    except Exception:
        return None


def _looks_like_tracks(array) -> bool:
    if array is None:
        return False

    if array.size == 0:
        return True

    if array.ndim == 1 and (array.size % 5 == 0 or array.size % 6 == 0):
        return True

    if array.ndim == 2 and array.shape[1] in (5, 6):
        return True

    if array.ndim == 2 and array.shape[1] >= 6:
        return True

    return False


def normalize_deepsort_outputs(outputs, debug: bool = False) -> np.ndarray:
    """
    Normalizes DeepSORT output to ndarray of shape (N, 6):

        [x1, y1, x2, y2, dummy, track_id]

    Different deep_sort_pytorch versions may return arrays, lists or tuples,
    so this helper keeps the tracking scripts robust.
    """
    if outputs is None:
        return np.empty((0, 6), dtype=np.float32)

    raw_outputs = outputs

    if isinstance(outputs, tuple):
        if debug:
            print(f"DeepSORT returned tuple(len={len(outputs)})")

            for index, part in enumerate(outputs):
                shape = getattr(part, "shape", None)
                size = getattr(part, "size", None)
                print(f"  part{index}: type={type(part)} shape={shape} size={size}")

        selected = None

        for part in outputs:
            array_part = _asarray_quiet(part)

            if _looks_like_tracks(array_part):
                selected = part
                break

        raw_outputs = selected if selected is not None else outputs[0]

    elif isinstance(outputs, list):
        if len(outputs) == 0:
            return np.empty((0, 6), dtype=np.float32)

        raw_outputs = outputs[0] if len(outputs) == 1 else outputs

    array = _asarray_quiet(raw_outputs)

    if array is None or array.size == 0:
        return np.empty((0, 6), dtype=np.float32)

    if array.ndim == 1:
        if array.size % 6 == 0:
            array = array.reshape(-1, 6)
        elif array.size % 5 == 0:
            array = array.reshape(-1, 5)
        else:
            if debug:
                print(f"DeepSORT: cannot reshape flat output of size {array.size}")

            return np.empty((0, 6), dtype=np.float32)

    if array.ndim == 2 and array.shape[1] == 5:
        normalized = np.zeros((array.shape[0], 6), dtype=np.float32)
        normalized[:, 0:4] = array[:, 0:4].astype(np.float32)
        normalized[:, 5] = array[:, 4].astype(np.float32)

        return normalized

    if array.ndim == 2 and array.shape[1] >= 6:
        return array[:, :6].astype(np.float32)

    return np.empty((0, 6), dtype=np.float32)