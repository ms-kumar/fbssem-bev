# Contributing to fbssem-bev

## Development setup

```bash
uv sync --group dev
```

## Standards

- Use Python 3.11-compatible syntax.
- Keep public APIs explicit and estimator-compatible.
- Use vectorized NumPy for numerical geometry and blending.
- Keep OpenCV at the image I/O, remapping, and inpainting boundary.
- Add type annotations to public and internal functions.
- Use NumPyDoc-style docstrings for public APIs.
- Keep lines at or below 88 characters.
- Use meaningful names and vectorized tensor operations.
- Raise specific exceptions with actionable messages.
- Do not add a runtime dependency without documenting why it is needed.

## Validation

```bash
uv run ruff check src tests
uv run pytest -q
```

Behavior changes must include focused tests. Dataset-independent synthetic
fixtures are preferred so CI does not require the FB-SSEM download.

## Release notes

Record user-visible changes in `RELEASES.md` under `Added`, `Changed`, `Fixed`,
`Deprecated`, `Removed`, or `Security`. Do not create per-version release-note
files.