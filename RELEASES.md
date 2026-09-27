# fbssem-bev Release Notes

## 0.1.0 - 2026-09-27

### Added

- Added a standalone uv package with a `src` layout.
- Added `FBSSEMBEVTransformer` with scikit-learn-style estimator methods.
- Added NumPy-based Mei projection geometry and corner blending.
- Added the `fbssem-bev` command for range-based and explicit-ID generation.
- Added cached rectilinear previews and configurable output directories.
- Added synthetic configuration, calibration, estimator, and CLI tests.
- Added package installation, architecture, API, and algorithm documentation.
- Added private-repository metadata, CI, security guidance, and result images.
- Added proprietary licensing that requires prior written permission.

### Changed

- Isolated FB-SSEM processing from the original `surround_view` and PyQt
  runtime.
- Standardized output on the official FB-SSEM `600x600` coordinate frame.

### Fixed

- Avoided the former rectification-plus-perspective double resampling path.
- Removed content-dependent contour generation from overlap blending.
- Filled the bounded ego blind region before masked vehicle compositing.

### Removed

- Removed the PyTorch runtime dependency in favor of vectorized NumPy.

### Security

- No security changes.

### Contributors

- Initial standalone package contributors.