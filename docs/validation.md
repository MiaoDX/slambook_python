# Validation Notes

Last validated: 2026-05-10

Validation uses the repo's `uv` environment plus sample files from upstream
`gaoxiang12/slambook` commit `853abac`.

The upstream sample smoke suite is repeatable with:

```bash
uv run --extra core --extra test --frozen python examples/reference/validate_upstream_samples.py \
  --upstream-root /path/to/gaoxiang12/slambook \
  --work-dir /tmp/slambook-python-validation
```

The command writes `/tmp/slambook-python-validation/upstream_validation_report.json`.

## Core Test Suite

```bash
uv run --extra core --extra test --frozen python -m pytest
```

Earlier recorded result: `132 passed`.

Latest local run on 2026-05-10:

```bash
uv run --extra core --extra test --frozen python -m pytest
```

Result: `139 passed in 4.70s`.

## Upstream Sample Smoke Tests

Chapter 7 3D-3D pose estimation on `ch7/1.png`, `ch7/2.png`, and depth maps:

- match count: `161`
- valid 3D-3D correspondence count: `150`
- registration RMSE: `0.118888236 m`

Chapter 9 depth-assisted local-map VO on the same RGB-D pair:

- inserted first-frame map points: `300`
- tracked second-frame PnP inliers: `57`
- final keyframes: `2`
- final map points: `600`

Chapter 12 BoW loop closure on upstream `ch12/data`:

- image count: `10`
- local descriptor count: `9993`
- vocabulary words: `8`
- BoW descriptor matrix: `(10, 8)`
- retrieval candidate count at index 9 with temporal window 1: `3`

Chapter 13 known-pose RGB-D fusion on upstream `ch13/dense_RGBD/data`:

- frame count: `5`
- fused point count before downsampling: `1081843`
- downsampled point count with `--voxel-size 0.02`: `293842`
- occupied voxel count with `--occupancy-voxel-size 0.05`: `65815`

## Benchmark Reports

Quantitative report JSON can be generated for trajectory, BAL, and pose-graph
examples with `examples/reference/benchmark_report.py`. For example:

```bash
uv run --extra core --extra test --frozen python examples/reference/benchmark_report.py \
  pose-graph \
  --g2o examples/ch11_pose_graph/tiny_pose_graph.g2o \
  --solve \
  --output /tmp/slambook-python-validation/pose_graph_report.json
```

## Optional Backend Checks

The all-extras environment was installed on macOS/Python 3.12 with the Tsinghua
uv mirror:

```bash
UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple uv sync --all-extras --frozen
```

Result: installed `74` optional packages after preparing `73` packages in
`63m 18s`.

The first plain optional test run aborted while importing `pycolmap` after
`faiss`, consistent with duplicate OpenMP runtime loading from optional native
wheels on macOS. The suite passed with the documented workaround:

```bash
KMP_DUPLICATE_LIB_OK=TRUE UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple uv run --all-extras --frozen python -m pytest tests_optional
```

Result: `8 passed, 3 warnings in 11.89s`.
