"""Configuration and calibration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from fbssem_bev._calibration import build_rectilinear_maps, load_mei_calibration
from fbssem_bev._config import BEVConfig


def test_config_uses_official_frame(synthetic_dataset: Path) -> None:
    config = BEVConfig.from_dataset_root(synthetic_dataset)

    assert (config.width, config.height) == (600, 600)
    assert (config.origin_u, config.origin_v) == (300.0, 333.0)
    assert config.car_shape == (112, 58)
    assert config.patch_bounds("front") == (0, 600, 0, 251)


def test_config_rejects_invalid_focal_scale(synthetic_dataset: Path) -> None:
    with pytest.raises(ValueError, match="preview_focal_scale"):
        BEVConfig.from_dataset_root(
            synthetic_dataset,
            preview_focal_scale=0,
        )


def test_load_and_build_rectilinear_maps(synthetic_dataset: Path) -> None:
    calibration = load_mei_calibration(
        synthetic_dataset / "yaml" / "camera_intrinsics.yaml"
    )

    map_x, map_y, virtual_intrinsic = build_rectilinear_maps(
        calibration,
        width=1280,
        height=1080,
        focal_scale=0.75,
    )

    assert calibration.intrinsic_matrix.shape == (3, 3)
    assert map_x.shape == (1080, 1280)
    assert map_y.shape == (1080, 1280)
    assert virtual_intrinsic[0, 0] == pytest.approx(495.0)