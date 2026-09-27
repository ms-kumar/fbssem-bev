"""Configuration and coordinate conventions for FB-SSEM projection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CameraPose:
    """A camera pose in the FB-SSEM vehicle coordinate system.

    Parameters
    ----------
    position : tuple of float
        Camera position ``(x, y, z)`` in meters.
    yaw_pitch : tuple of float
        Camera yaw and pitch in degrees.
    """

    position: tuple[float, float, float]
    yaw_pitch: tuple[float, float]


CAMERA_POSES: dict[str, CameraPose] = {
    "front": CameraPose((0.0, 0.406, 3.873), (0.0, 26.0)),
    "back": CameraPose((0.132, 0.744, -1.001), (180.0, 3.0)),
    "left": CameraPose((-1.024, 0.8, 2.053), (-90.0, 0.0)),
    "right": CameraPose((1.015, 0.801, 2.04), (90.0, 0.0)),
}

CAMERA_FOLDERS: dict[str, str] = {
    "front": "front",
    "back": "rear",
    "left": "left",
    "right": "right",
}


@dataclass(frozen=True)
class BEVConfig:
    """Runtime configuration for the FB-SSEM BEV transformer.

    Parameters
    ----------
    dataset_root : pathlib.Path
        Root containing ``rgb``, ``seg``, and ``yaml`` directories.
    output_dir : pathlib.Path
        Destination for generated BEV images.
    preview_dir : pathlib.Path
        Destination for rectilinear camera previews.
    preview_focal_scale : float, default=0.75
        Focal-length scale for rectilinear diagnostic previews.
    """

    dataset_root: Path
    output_dir: Path
    preview_dir: Path
    preview_focal_scale: float = 0.75
    source_width: int = 1280
    source_height: int = 1080
    width: int = 600
    height: int = 600
    pixels_per_meter: float = 24.0
    origin_u: float = 300.0
    origin_v: float = 333.0
    car_left: int = 271
    car_right: int = 329
    car_top: int = 251
    car_bottom: int = 363
    reference_id: str = "0"
    tone_gains: tuple[float, float, float] = (0.63, 0.68, 0.70)

    def __post_init__(self) -> None:
        if self.preview_focal_scale <= 0:
            raise ValueError("preview_focal_scale must be positive")
        if self.source_width <= 0 or self.source_height <= 0:
            raise ValueError("source dimensions must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("BEV dimensions must be positive")
        if self.pixels_per_meter <= 0:
            raise ValueError("pixels_per_meter must be positive")
        if not (0 <= self.car_left < self.car_right <= self.width):
            raise ValueError("car horizontal bounds are outside the BEV frame")
        if not (0 <= self.car_top < self.car_bottom <= self.height):
            raise ValueError("car vertical bounds are outside the BEV frame")

    @classmethod
    def from_dataset_root(
        cls,
        dataset_root: str | Path,
        *,
        output_dir: str | Path | None = None,
        preview_dir: str | Path | None = None,
        preview_focal_scale: float = 0.75,
    ) -> BEVConfig:
        """Build a configuration using the package's standard directories."""

        root = Path(dataset_root).expanduser().resolve()
        output = (
            Path(output_dir).expanduser().resolve()
            if output_dir
            else root / "bev_output"
        )
        preview = (
            Path(preview_dir).expanduser().resolve()
            if preview_dir
            else root / "rgb_und"
        )
        return cls(
            dataset_root=root,
            output_dir=output,
            preview_dir=preview,
            preview_focal_scale=preview_focal_scale,
        )

    @property
    def car_shape(self) -> tuple[int, int]:
        """Return the ego footprint as ``(height, width)``."""

        return self.car_bottom - self.car_top, self.car_right - self.car_left

    def patch_bounds(self, camera_name: str) -> tuple[int, int, int, int]:
        """Return ``(u0, u1, v0, v1)`` for a camera's output patch."""

        bounds = {
            "front": (0, self.width, 0, self.car_top),
            "back": (0, self.width, self.car_bottom, self.height),
            "left": (0, self.car_left, 0, self.height),
            "right": (self.car_right, self.width, 0, self.height),
        }
        try:
            return bounds[camera_name]
        except KeyError as error:
            raise ValueError(f"unknown camera: {camera_name}") from error