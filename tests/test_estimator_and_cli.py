"""Estimator and command-line integration tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from fbssem_bev import FBSSEMBEVTransformer
from fbssem_bev.cli import main


def test_transform_requires_fit(synthetic_dataset: Path) -> None:
    transformer = FBSSEMBEVTransformer(synthetic_dataset)

    with pytest.raises(RuntimeError, match="call fit"):
        transformer.transform([0])


def test_estimator_transforms_and_writes_files(
    synthetic_dataset: Path,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    preview_dir = tmp_path / "previews"
    transformer = FBSSEMBEVTransformer(
        synthetic_dataset,
        output_dir=output_dir,
        preview_dir=preview_dir,
    ).fit()

    images = transformer.transform([0])
    paths = transformer.transform_to_files([0])

    assert transformer.n_features_in_ == 4
    assert images[0].shape == (600, 600, 3)
    assert images[0].dtype == np.uint8
    assert paths == [output_dir / "0.png"]
    assert cv2.imread(str(paths[0])).shape == (600, 600, 3)
    assert len(list(preview_dir.glob("*/*.png"))) == 4


def test_estimator_parameter_api(synthetic_dataset: Path) -> None:
    transformer = FBSSEMBEVTransformer(synthetic_dataset).fit()

    assert transformer.get_params()["preview_focal_scale"] == 0.75
    transformer.set_params(preview_focal_scale=0.8)
    assert not hasattr(transformer, "is_fitted_")

    with pytest.raises(ValueError, match="invalid parameter"):
        transformer.set_params(unknown=True)


def test_cli_writes_selected_sample(
    synthetic_dataset: Path,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "cli-output"
    preview_dir = tmp_path / "cli-previews"

    exit_code = main(
        [
            str(synthetic_dataset),
            "--ids",
            "0",
            "--output-dir",
            str(output_dir),
            "--preview-dir",
            str(preview_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "0.png").is_file()