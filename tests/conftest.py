"""Shared synthetic FB-SSEM test fixtures."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create the minimum valid FB-SSEM directory tree for one sample."""

    root = tmp_path_factory.mktemp("fb_ssem")
    camera_colors = {
        "front": (20, 40, 80),
        "rear": (30, 60, 90),
        "left": (40, 80, 120),
        "right": (50, 100, 150),
    }
    for camera, color in camera_colors.items():
        camera_dir = root / "rgb" / camera
        camera_dir.mkdir(parents=True)
        image = np.full((1080, 1280, 3), color, dtype=np.uint8)
        assert cv2.imwrite(str(camera_dir / "0.png"), image)

    bev_dir = root / "rgb" / "bev"
    seg_dir = root / "seg" / "bev"
    bev_dir.mkdir(parents=True)
    seg_dir.mkdir(parents=True)
    reference = np.full((600, 600, 3), 70, dtype=np.uint8)
    segmentation = np.zeros((600, 600, 3), dtype=np.uint8)
    reference[251:363, 271:329] = (220, 220, 220)
    segmentation[251:363, 271:329] = (255, 255, 255)
    assert cv2.imwrite(str(bev_dir / "0.png"), reference)
    assert cv2.imwrite(str(seg_dir / "0.png"), segmentation)

    yaml_dir = root / "yaml"
    yaml_dir.mkdir()
    storage = cv2.FileStorage(
        str(yaml_dir / "camera_intrinsics.yaml"),
        cv2.FILE_STORAGE_WRITE,
    )
    storage.write(
        "K",
        np.array(
            [[660.0, -2.8, 634.0], [0.0, 625.0, 544.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
    )
    storage.write("D", np.array([[-0.29, 0.11, 0.0, 0.003]], dtype=np.float64))
    storage.write("xi", np.array([[1.086]], dtype=np.float64))
    storage.release()
    return root