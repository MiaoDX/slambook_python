# Porting Rules

The migration ports slambook examples by preserving the lesson, not by
mechanically translating C++ syntax.

## Baseline First

- Start each chapter with NumPy, SciPy, and OpenCV implementations.
- Keep optional modern backends behind extras and import guards.
- Do not require GPU, GTSAM, PyCOLMAP, LightGlue, FAISS, JAX, or Rerun to import
  `slam` or run core tests.

## Public API Shape

- Use `Nx2` image points and `Nx3` 3D points at public boundaries.
- Convert OpenCV masks with `mask.ravel() != 0`; OpenCV may return either `1`
  or `255` for inliers depending on the function.
- Use coordinate names from `docs/coordinates.md`.
- Prefer explicit result dataclasses over returning long unlabelled tuples.

## Examples

- Keep examples executable from the command line.
- Use image, dataset, and output paths from CLI arguments.
- Print structured numerical results that can be inspected or copied into tests.
- Avoid hard-coded local Windows, macOS, or Linux user paths.

## Legacy Scripts

Root-level scripts remain as the legacy baseline during the first migration
wave. New implementations should live under `slam/` and `examples/`; once the
replacement examples are stable, root scripts can become wrappers or move under
`legacy/` with documentation.
