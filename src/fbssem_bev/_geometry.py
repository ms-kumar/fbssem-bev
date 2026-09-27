"""NumPy geometry for direct FB-SSEM ground-to-fisheye projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._calibration import MeiCalibration
from ._config import CAMERA_POSES, BEVConfig

FloatMap: TypeAlias = NDArray[np.float32]
BoolMap: TypeAlias = NDArray[np.bool_]


@dataclass(frozen=True)
class ProjectionMap:
    """OpenCV remap coordinates and their geometric validity mask."""

    x: FloatMap
    y: FloatMap
    valid: BoolMap


def _world_to_camera(camera_name: str) -> NDArray[np.float64]:
    pose = CAMERA_POSES[camera_name]
    yaw, pitch = np.deg2rad(pose.yaw_pitch)
    cosine_yaw, sine_yaw = np.cos(yaw), np.sin(yaw)
    cosine_pitch, sine_pitch = np.cos(pitch), np.sin(pitch)
    yaw_rotation = np.array(
        [
            [cosine_yaw, 0.0, sine_yaw],
            [0.0, 1.0, 0.0],
            [-sine_yaw, 0.0, cosine_yaw],
        ],
        dtype=np.float64,
    )
    pitch_rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cosine_pitch, -sine_pitch],
            [0.0, sine_pitch, cosine_pitch],
        ],
        dtype=np.float64,
    )
    invert_y = np.diag(np.array((1.0, -1.0, 1.0), dtype=np.float64))
    return invert_y @ (yaw_rotation @ pitch_rotation).T


def build_ground_projection_map(
    calibration: MeiCalibration,
    config: BEVConfig,
    camera_name: str,
) -> ProjectionMap:
    """Build a direct inverse map from a BEV patch to a raw fisheye image."""

    if camera_name not in CAMERA_POSES:
        raise ValueError(f"unknown camera: {camera_name}")

    intrinsic = calibration.intrinsic_matrix
    u0, u1, v0, v1 = config.patch_bounds(camera_name)
    v_coordinates, u_coordinates = np.mgrid[v0:v1, u0:u1].astype(np.float64)
    ground_x = (u_coordinates - config.origin_u) / config.pixels_per_meter
    ground_z = (config.origin_v - v_coordinates) / config.pixels_per_meter

    position = CAMERA_POSES[camera_name].position
    relative_points = np.stack(
        (
            ground_x - position[0],
            np.full_like(ground_x, -position[1]),
            ground_z - position[2],
        )
    )
    camera_points = np.einsum(
        "ij,jhw->ihw",
        _world_to_camera(camera_name),
        relative_points,
    )
    camera_x, camera_y, camera_z = camera_points
    norm = np.linalg.norm(camera_points, axis=0)
    denominator = camera_z + calibration.xi * norm
    valid_denominator = denominator > 1e-9
    normalized_x = np.divide(
        camera_x,
        denominator,
        out=np.zeros_like(camera_x),
        where=valid_denominator,
    )
    normalized_y = np.divide(
        camera_y,
        denominator,
        out=np.zeros_like(camera_y),
        where=valid_denominator,
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
    )
    map_y = intrinsic[1, 1] * distorted_y + intrinsic[1, 2]
    valid = (
        (camera_z > 0)
        & valid_denominator
        & (map_x >= 0)
        & (map_x < config.source_width - 1)
        & (map_y >= 0)
        & (map_y < config.source_height - 1)
    )
    return ProjectionMap(
        x=map_x.astype(np.float32),
        y=map_y.astype(np.float32),
        valid=valid,
    )


def build_blend_weight(
    width: int,
    height: int,
    corner: str,
    *,
    band: float = 0.12,
) -> FloatMap:
    """Build a three-channel diagonal blend ramp for one overlap corner."""

    if width <= 0 or height <= 0:
        raise ValueError("blend dimensions must be positive")
    if band <= 0:
        raise ValueError("band must be positive")

    y_coordinates, x_coordinates = np.mgrid[0:height, 0:width].astype(np.float32)
    normalized_x = (x_coordinates + 0.5) / width
    normalized_y = (y_coordinates + 0.5) / height
    distances = {
        "front_left": normalized_x - normalized_y,
        "front_right": 1 - normalized_x - normalized_y,
        "back_left": normalized_x + normalized_y - 1,
        "back_right": normalized_y - normalized_x,
    }
    try:
        distance = distances[corner]
    except KeyError as error:
        raise ValueError(f"unknown corner: {corner}") from error
    weight = np.clip(0.5 + distance / (2 * band), 0, 1)
    return np.repeat(weight[:, :, None], 3, axis=2)


def build_blend_weights(config: BEVConfig) -> dict[str, FloatMap]:
    """Build all four overlap weights."""

    return {
        "front_left": build_blend_weight(
            config.car_left,
            config.car_top,
            "front_left",
        ),
        "front_right": build_blend_weight(
            config.width - config.car_right,
            config.car_top,
            "front_right",
        ),
        "back_left": build_blend_weight(
            config.car_left,
            config.height - config.car_bottom,
            "back_left",
        ),
        "back_right": build_blend_weight(
            config.width - config.car_right,
            config.height - config.car_bottom,
            "back_right",
        ),
    }