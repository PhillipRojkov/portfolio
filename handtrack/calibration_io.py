"""Minimal helpers for sharing fisheye calibration values via JSON.

This module is intentionally standalone so it can be imported from anywhere
without modifying camera_calibration.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DEFAULT_VALUES_PATH = Path(__file__).with_name("calibration_values.json")


@dataclass
class FisheyeParams:
    k: np.ndarray
    d: np.ndarray
    image_size: tuple[int, int] | None = None  # (width, height)


def load_fisheye_params(path: str | Path = DEFAULT_VALUES_PATH) -> FisheyeParams:
    """Load K and D from a calibration JSON file."""

    file_path = Path(path)
    data = json.loads(file_path.read_text(encoding="utf-8"))

    k = np.array(data["K"], dtype=np.float64)
    d = np.array(data["D"], dtype=np.float64).reshape(4, 1)

    if k.shape != (3, 3):
        raise ValueError("K must be shape [3, 3].")

    image_size = None
    if "image_width" in data and "image_height" in data:
        image_size = (int(data["image_width"]), int(data["image_height"]))

    return FisheyeParams(k=k, d=d, image_size=image_size)


def save_fisheye_params(params: FisheyeParams, path: str | Path = DEFAULT_VALUES_PATH) -> None:
    """Save K and D to a calibration JSON file."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": "fisheye",
        "K": params.k.astype(float).tolist(),
        "D": params.d.astype(float).reshape(4, 1).tolist(),
    }
    if params.image_size is not None:
        payload["image_width"] = int(params.image_size[0])
        payload["image_height"] = int(params.image_size[1])

    file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_undistort_maps(
    frame_width: int,
    frame_height: int,
    params: FisheyeParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Build OpenCV remap matrices for fisheye undistortion."""

    return cv2.fisheye.initUndistortRectifyMap(
        params.k,
        params.d,
        np.eye(3),
        params.k,
        (int(frame_width), int(frame_height)),
        cv2.CV_16SC2,
    )


def undistort_frame(frame: np.ndarray, params: FisheyeParams | None = None) -> np.ndarray:
    """Undistort a frame using loaded fisheye params.

    If params is None, calibration values are loaded from DEFAULT_VALUES_PATH.
    """

    if params is None:
        params = load_fisheye_params()

    h, w = frame.shape[:2]
    map1, map2 = build_undistort_maps(w, h, params)
    return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
