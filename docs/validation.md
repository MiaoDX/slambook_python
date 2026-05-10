# Validation Notes

Last validated: 2026-05-10

Validation uses the repo's `uv` environment plus sample files from upstream
`gaoxiang12/slambook` commit `853abac`.

## Core Test Suite

```bash
uv run --extra core --extra test --frozen python -m pytest
```

Result: `132 passed`.

## Upstream Sample Smoke Tests

Chapter 7 3D-3D pose estimation on `ch7/1.png`, `ch7/2.png`, and depth maps:

- match count: `161`
- valid 3D-3D correspondence count: `150`
- registration RMSE: `0.120947847 m`

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

Chapter 13 known-pose RGB-D fusion on upstream `ch13/dense_RGBD/data`:

- frame count: `5`
- fused point count before downsampling: `1081843`
- downsampled point count with `--voxel-size 0.02`: `33841`
- occupied voxel count with `--occupancy-voxel-size 0.05`: `5596`
