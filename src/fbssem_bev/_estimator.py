"""Scikit-learn-style FB-SSEM bird's-eye-view transformer."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

from ._calibration import build_rectilinear_maps, load_mei_calibration
from ._config import CAMERA_FOLDERS, BEVConfig
from ._geometry import (
    ProjectionMap,
    build_blend_weights,
    build_ground_projection_map,
)

Image: TypeAlias = NDArray[np.uint8]


class FBSSEMBEVTransformer:
    """Transform FB-SSEM surround-camera samples into stitched BEV images.

    Parameters
    ----------
    dataset_root : str or pathlib.Path
        Root containing the FB-SSEM ``rgb``, ``seg``, and ``yaml`` folders.
    output_dir : str or pathlib.Path, optional
        BEV destination. Defaults to ``<dataset_root>/bev_output``.
    preview_dir : str or pathlib.Path, optional
        Rectilinear preview destination. Defaults to
        ``<dataset_root>/rgb_und``.
    preview_focal_scale : float, default=0.75
        Virtual pinhole focal scale used only for diagnostic previews.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        output_dir: str | Path | None = None,
        preview_dir: str | Path | None = None,
        preview_focal_scale: float = 0.75,
    ) -> None:
        self.dataset_root = dataset_root
        self.output_dir = output_dir
        self.preview_dir = preview_dir
        self.preview_focal_scale = preview_focal_scale

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return constructor parameters for estimator compatibility."""

        del deep
        return {
            "dataset_root": self.dataset_root,
            "output_dir": self.output_dir,
            "preview_dir": self.preview_dir,
            "preview_focal_scale": self.preview_focal_scale,
        }

    def set_params(self, **params: Any) -> FBSSEMBEVTransformer:
        """Set constructor parameters and invalidate fitted state."""

        valid_params = self.get_params()
        unknown = sorted(set(params) - set(valid_params))
        if unknown:
            raise ValueError(f"invalid parameter(s): {', '.join(unknown)}")
        for name, value in params.items():
            setattr(self, name, value)
        self._clear_fitted_state()
        return self

    def fit(
        self,
        X: object | None = None,
        y: object | None = None,
    ) -> FBSSEMBEVTransformer:
        """Load calibration and precompute all static projection maps."""

        del X, y
        self.config_ = BEVConfig.from_dataset_root(
            self.dataset_root,
            output_dir=self.output_dir,
            preview_dir=self.preview_dir,
            preview_focal_scale=self.preview_focal_scale,
        )
        self._validate_dataset()
        self.calibration_ = load_mei_calibration(
            self.config_.dataset_root / "yaml" / "camera_intrinsics.yaml",
        )
        (
            self.preview_map_x_,
            self.preview_map_y_,
            self.virtual_intrinsic_,
        ) = build_rectilinear_maps(
            self.calibration_,
            width=self.config_.source_width,
            height=self.config_.source_height,
            focal_scale=self.config_.preview_focal_scale,
        )
        self.projection_maps_ = {
            name: build_ground_projection_map(
                self.calibration_, self.config_, name
            )
            for name in CAMERA_FOLDERS
        }
        self.blend_weights_ = build_blend_weights(self.config_)
        self.car_image_, self.car_mask_ = self._load_ego_sprite()
        self.ego_fill_mask_ = self._build_ego_fill_mask()
        self.config_.output_dir.mkdir(parents=True, exist_ok=True)
        self.config_.preview_dir.mkdir(parents=True, exist_ok=True)
        self.n_features_in_ = 4
        self.is_fitted_ = True
        return self

    def transform(self, X: Iterable[int | str]) -> list[Image]:
        """Transform sample IDs into BEV images without writing BEV files."""

        self._check_is_fitted()
        return [self._transform_one(str(sample_id)) for sample_id in X]

    def fit_transform(
        self,
        X: Iterable[int | str],
        y: object | None = None,
    ) -> list[Image]:
        """Fit static geometry, then transform sample IDs."""

        return self.fit(X, y).transform(X)

    def transform_to_files(self, X: Iterable[int | str]) -> list[Path]:
        """Transform sample IDs and write PNG files to ``output_dir``."""

        self._check_is_fitted()
        output_paths: list[Path] = []
        for sample_id in X:
            normalized_id = str(sample_id)
            image = self._transform_one(normalized_id)
            output_path = self.config_.output_dir / f"{normalized_id}.png"
            if not cv2.imwrite(str(output_path), image):
                raise OSError(f"failed to write BEV image: {output_path}")
            output_paths.append(output_path)
        return output_paths

    def available_sample_ids(self) -> list[str]:
        """Return numeric front-camera sample IDs in ascending order."""

        root = Path(self.dataset_root).expanduser().resolve()
        front_dir = root / "rgb" / "front"
        if not front_dir.is_dir():
            raise FileNotFoundError(front_dir)
        sample_ids = [path.stem for path in front_dir.glob("*.png")]
        try:
            return sorted(sample_ids, key=int)
        except ValueError:
            return sorted(sample_ids)

    def _clear_fitted_state(self) -> None:
        for name in tuple(vars(self)):
            if name.endswith("_"):
                delattr(self, name)

    def _check_is_fitted(self) -> None:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("call fit before transform")

    def _validate_dataset(self) -> None:
        required = [
            self.config_.dataset_root / "rgb" / folder
            for folder in (*CAMERA_FOLDERS.values(), "bev")
        ]
        required.extend(
            (
                self.config_.dataset_root / "seg" / "bev",
                self.config_.dataset_root / "yaml" / "camera_intrinsics.yaml",
            )
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "missing required FB-SSEM paths:\n" + "\n".join(missing)
            )

    def _load_ego_sprite(self) -> tuple[Image, NDArray[np.bool_]]:
        sample_id = self.config_.reference_id
        reference_path = self.config_.dataset_root / "rgb" / "bev" / f"{sample_id}.png"
        segmentation_path = (
            self.config_.dataset_root / "seg" / "bev" / f"{sample_id}.png"
        )
        reference = cv2.imread(str(reference_path))
        segmentation = cv2.imread(str(segmentation_path))
        if reference is None:
            raise FileNotFoundError(reference_path)
        if segmentation is None:
            raise FileNotFoundError(segmentation_path)
        if reference.shape[:2] != (self.config_.height, self.config_.width):
            raise ValueError("reference BEV dimensions do not match configuration")

        top, bottom = self.config_.car_top, self.config_.car_bottom
        left, right = self.config_.car_left, self.config_.car_right
        car_image = reference[top:bottom, left:right].copy()
        car_mask = np.all(segmentation[top:bottom, left:right] > 220, axis=2)
        if not np.any(car_mask):
            raise ValueError("ego mask is empty in the reference segmentation")
        return car_image, car_mask

    def _build_ego_fill_mask(self) -> NDArray[np.uint8]:
        mask = np.zeros(
            (self.config_.height, self.config_.width), dtype=np.uint8
        )
        top = max(0, self.config_.car_top - 12)
        bottom = min(self.config_.height, self.config_.car_bottom + 4)
        left = max(0, self.config_.car_left - 4)
        right = min(self.config_.width, self.config_.car_right + 4)
        mask[top:bottom, left:right] = 255
        return mask

    def _read_raw_image(self, camera_name: str, sample_id: str) -> Image:
        folder = CAMERA_FOLDERS[camera_name]
        path = self.config_.dataset_root / "rgb" / folder / f"{sample_id}.png"
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        expected_shape = (self.config_.source_height, self.config_.source_width)
        if image.shape[:2] != expected_shape:
            raise ValueError(
                f"{path} has shape {image.shape[:2]}, expected {expected_shape}"
            )
        return image

    def _ensure_preview(
        self,
        camera_name: str,
        sample_id: str,
        raw: Image,
    ) -> None:
        folder = CAMERA_FOLDERS[camera_name]
        source_path = (
            self.config_.dataset_root / "rgb" / folder / f"{sample_id}.png"
        )
        destination = self.config_.preview_dir / folder / f"{sample_id}.png"
        if (
            destination.is_file()
            and destination.stat().st_mtime >= source_path.stat().st_mtime
        ):
            return
        preview = cv2.remap(
            raw,
            self.preview_map_x_,
            self.preview_map_y_,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), preview):
            raise OSError(f"failed to write preview image: {destination}")

    def _project_camera(self, camera_name: str, sample_id: str) -> Image:
        raw = self._read_raw_image(camera_name, sample_id)
        self._ensure_preview(camera_name, sample_id, raw)
        projection: ProjectionMap = self.projection_maps_[camera_name]
        patch = cv2.remap(
            raw,
            projection.x,
            projection.y,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
        )
        patch[~projection.valid] = 0
        return patch

    def _blend(self, first: Image, second: Image, corner: str) -> Image:
        weight = self.blend_weights_[corner]
        blended = (
            first.astype(np.float32) * weight
            + second.astype(np.float32) * (1 - weight)
        )
        return blended.astype(np.uint8)

    def _stitch(self, patches: dict[str, Image]) -> Image:
        config = self.config_
        front = patches["front"]
        back = patches["back"]
        left = patches["left"]
        right = patches["right"]
        output = np.zeros((config.height, config.width, 3), dtype=np.uint8)

        output[: config.car_top, config.car_left : config.car_right] = front[
            :, config.car_left : config.car_right
        ]
        output[config.car_bottom :, config.car_left : config.car_right] = back[
            :, config.car_left : config.car_right
        ]
        output[config.car_top : config.car_bottom, : config.car_left] = left[
            config.car_top : config.car_bottom
        ]
        output[config.car_top : config.car_bottom, config.car_right :] = right[
            config.car_top : config.car_bottom
        ]
        output[: config.car_top, : config.car_left] = self._blend(
            front[:, : config.car_left],
            left[: config.car_top],
            "front_left",
        )
        output[: config.car_top, config.car_right :] = self._blend(
            front[:, config.car_right :],
            right[: config.car_top],
            "front_right",
        )
        output[config.car_bottom :, : config.car_left] = self._blend(
            back[:, : config.car_left],
            left[config.car_bottom :],
            "back_left",
        )
        output[config.car_bottom :, config.car_right :] = self._blend(
            back[:, config.car_right :],
            right[config.car_bottom :],
            "back_right",
        )
        return output

    def _apply_tone_gains(self, image: Image) -> Image:
        gains = np.asarray(self.config_.tone_gains, dtype=np.float32).reshape(
            1, 1, 3
        )
        return np.clip(image.astype(np.float32) * gains, 0, 255).astype(np.uint8)

    def _transform_one(self, sample_id: str) -> Image:
        patches = {
            name: self._project_camera(name, sample_id)
            for name in CAMERA_FOLDERS
        }
        output = self._apply_tone_gains(self._stitch(patches))
        output = cv2.inpaint(
            output,
            self.ego_fill_mask_,
            5,
            cv2.INPAINT_TELEA,
        )
        top, bottom = self.config_.car_top, self.config_.car_bottom
        left, right = self.config_.car_left, self.config_.car_right
        np.copyto(
            output[top:bottom, left:right],
            self.car_image_,
            where=self.car_mask_[:, :, None],
        )
        return output