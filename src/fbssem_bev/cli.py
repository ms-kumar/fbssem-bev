"""Command-line interface for FB-SSEM BEV generation."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

from ._estimator import FBSSEMBEVTransformer


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(
        prog="fbssem-bev",
        description="Generate 600x600 BEV images from FB-SSEM fisheye cameras.",
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="FB-SSEM root containing rgb, seg, and yaml directories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="BEV output directory (default: <dataset_root>/bev_output)",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        help="rectilinear preview directory (default: <dataset_root>/rgb_und)",
    )
    parser.add_argument("--start", type=int, default=0, help="inclusive start ID")
    parser.add_argument("--end", type=int, help="exclusive end ID")
    parser.add_argument(
        "--ids",
        nargs="+",
        help="explicit sample IDs; overrides --start and --end",
    )
    parser.add_argument(
        "--preview-focal-scale",
        type=float,
        default=0.75,
        help="rectilinear preview focal scale (default: 0.75)",
    )
    return parser


def _select_sample_ids(
    transformer: FBSSEMBEVTransformer,
    *,
    explicit_ids: list[str] | None,
    start: int,
    end: int | None,
) -> list[str]:
    if explicit_ids:
        return explicit_ids
    if start < 0:
        raise ValueError("start must be non-negative")

    available = transformer.available_sample_ids()
    selected = [sample_id for sample_id in available if int(sample_id) >= start]
    if end is not None:
        if end <= start:
            raise ValueError("end must be greater than start")
        selected = [sample_id for sample_id in selected if int(sample_id) < end]
    if not selected:
        raise ValueError("no samples selected")
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    """Run the FB-SSEM BEV command-line workflow."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        transformer = FBSSEMBEVTransformer(
            args.dataset_root,
            output_dir=args.output_dir,
            preview_dir=args.preview_dir,
            preview_focal_scale=args.preview_focal_scale,
        )
        sample_ids = _select_sample_ids(
            transformer,
            explicit_ids=args.ids,
            start=args.start,
            end=args.end,
        )
        print("fitting static geometry...", flush=True)
        transformer.fit()
        started = time.perf_counter()
        for index, sample_id in enumerate(sample_ids, start=1):
            transformer.transform_to_files([sample_id])
            if index % 10 == 0 or index == len(sample_ids):
                elapsed = time.perf_counter() - started
                print(
                    f"[{index}/{len(sample_ids)}] elapsed={elapsed:.1f}s",
                    flush=True,
                )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")

    print(f"wrote {len(sample_ids)} image(s) to {transformer.config_.output_dir}")
    return 0