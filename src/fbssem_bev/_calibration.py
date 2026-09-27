"""Mei omnidirectional camera calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

FloatMap: TypeAlias = NDArray[np.float32]


@dataclass(frozen=True)
class MeiCalibration:
    """FB-SSEM Mei camera parameters stored as NumPy arrays."""

    intrinsic_matrix: NDArray[np.float64]
    distortion: NDArray[np.float64]
    xi: float


def _read_matrix(storage: cv2.FileStorage, key: str) -> NDArray[np.float64]:
    node = storage.getNode(key)
    if node.empty():
        raise ValueError(f"missing calibration matrix: {key}")
    matrix = node.mat()
    if matrix is None:
        raise ValueError(f"invalid calibration matrix: {key}")
    return np.asarray(matrix, dtype=np.float64)


def load_mei_calibration(
    path: str | Path,
) -> MeiCalibration:
    """Load FB-SSEM Mei parameters from an OpenCV YAML file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to ``camera_intrinsics.yaml``.

    Returns
    -------
    calibration : MeiCalibration
        Validated intrinsic matrix, distortion vector, and mirror parameter.
    """

    calibration_path = Path(path)
    if not calibration_path.is_file():
        raise FileNotFoundError(calibration_path)

    storage = cv2.FileStorage(str(calibration_path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise ValueError(f"cannot open calibration file: {calibration_path}")
    try:
        intrinsic = _read_matrix(storage, "K")
        distortion = _read_matrix(storage, "D").reshape(-1)
        xi_values = _read_matrix(storage, "xi").reshape(-1)
    finally:
        storage.release()

    if intrinsic.shape != (3, 3):
        raise ValueError(f"K must have shape (3, 3), got {intrinsic.shape}")
    if distortion.size < 4:
        raise ValueError("D must contain at least four coefficients")
    if xi_values.size != 1:
        raise ValueError("xi must contain exactly one value")

    return MeiCalibration(
        intrinsic_matrix=intrinsic,
        distortion=distortion[:4],
        xi=float(xi_values[0]),
    )


def build_rectilinear_maps(
    calibration: MeiCalibration,
    *,
    width: int,
    height: int,
    focal_scale: float,
) -> tuple[FloatMap, FloatMap, NDArray[np.float64]]:
    """Build inverse Mei maps for a rectilinear diagnostic preview.

    Returns
    -------
    map_x, map_y : numpy.ndarray of float32
        OpenCV remap coordinates into the raw fisheye image.
    virtual_intrinsic : numpy.ndarray of float64
        Intrinsic matrix of the rectilinear preview camera.
    """

    if width <= 0 or height <= 0:
        raise ValueError("preview dimensions must be positive")
    if focal_scale <= 0:
        raise ValueError("focal_scale must be positive")

    intrinsic = calibration.intrinsic_matrix
    virtual_intrinsic = intrinsic.copy()
    virtual_intrinsic[0, 0] *= focal_scale
    virtual_intrinsic[0, 1] *= focal_scale
    virtual_intrinsic[1, 1] *= focal_scale

    y_coordinates, x_coordinates = np.mgrid[0:height, 0:width].astype(np.float64)
    homogeneous = np.stack(
        (x_coordinates, y_coordinates, np.ones_like(x_coordinates))
    ).reshape(3, -1)
    rays = np.linalg.inv(virtual_intrinsic) @ homogeneous
    rays /= np.linalg.norm(rays, axis=0, keepdims=True)

    denominator = rays[2] + calibration.xi
    valid = denominator > 1e-9
    normalized_x = np.divide(
        rays[0], denominator, out=np.zeros_like(rays[0]), where=valid
    )
    normalized_y = np.divide(
        rays[1], denominator, out=np.zeros_like(rays[1]), where=valid
    )

    k1, k2, p1, p2 = calibration.distortion
    radius_squared = normalized_x**2 + normalized_y**2
    radial = 1 + k1 * radius_squared + k2 * radius_squared**2
    distorted_x = (
        normalized_x * radial
        + 2 * p1 * normalized_x * normalized_y
        + p2 * (radius_squared + 2 * normalized_x**2)
    )
    distorted_y = (
        normalized_y * radial
        + p1 * (radius_squared + 2 * normalized_y**2)
        + 2 * p2 * normalized_x * normalized_y
    )

    map_x = (
        intrinsic[0, 0] * distorted_x
        + intrinsic[0, 1] * distorted_y
        + intrinsic[0, 2]
    ).reshape(height, width)
    map_y = (intrinsic[1, 1] * distorted_y + intrinsic[1, 2]).reshape(
        height, width
    )
    return (
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        virtual_intrinsic,
    )