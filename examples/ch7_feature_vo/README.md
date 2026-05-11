# Chapter 7 Feature Visual Odometry

This directory contains command-line replacements for the legacy root scripts.

## 2D-2D Pose Estimation

```bash
uv run --frozen python examples/ch7_feature_vo/pose_estimation_2d2d.py \
  --image0 data/slambook/ch7/1.png \
  --image1 data/slambook/ch7/2.png \
  --matcher orb
```

Use explicit intrinsics when they are known:

```bash
uv run --frozen python examples/ch7_feature_vo/pose_estimation_2d2d.py \
  --image0 data/slambook/ch7/1.png \
  --image1 data/slambook/ch7/2.png \
  --matcher sift \
  --intrinsics 520.9 521.0 325.1 249.7
```

If no intrinsics are passed, the example uses a simple image-size-derived camera
matrix so the pipeline remains runnable for inspection. Pose estimates from that
fallback should not be treated as calibrated results.

## Triangulation

```bash
uv run --frozen python examples/ch7_feature_vo/triangulation.py \
  --image0 data/slambook/ch7/1.png \
  --image1 data/slambook/ch7/2.png \
  --matcher orb \
  --intrinsics 520.9 521.0 325.1 249.7 \
  --output-points outputs/ch7_points.npy
```

## Matcher Comparison

```bash
uv run --frozen python examples/ch7_feature_vo/compare_matchers.py \
  --image0 data/slambook/ch7/1.png \
  --image1 data/slambook/ch7/2.png \
  --matchers orb sift lightglue loftr
```

`lightglue` and `loftr` require the optional `modern` dependencies. They keep
the same matched-point interface as the OpenCV matchers, so downstream pose
examples can compare matcher quality without changing geometry code.

## 3D-2D Pose Estimation

`pose_estimation_3d2d.py` uses the first image's depth map to back-project
matched pixels into camera-frame 3D points, then estimates the pose of the
second image with PnP.

```bash
uv run --frozen python examples/ch7_feature_vo/pose_estimation_3d2d.py \
  --image0 data/slambook/ch7/1.png \
  --image1 data/slambook/ch7/2.png \
  --depth0 data/slambook/ch7/1_depth.png \
  --matcher orb \
  --intrinsics 520.9 521.0 325.1 249.7 \
  --depth-scale 5000
```

## 3D-3D Pose Estimation

`pose_estimation_3d3d.py` uses both depth maps to back-project matched pixels
into paired 3D camera-frame points, then estimates `T_10` with SVD rigid
registration.

```bash
uv run --frozen python examples/ch7_feature_vo/pose_estimation_3d3d.py \
  --image0 data/slambook/ch7/1.png \
  --image1 data/slambook/ch7/2.png \
  --depth0 data/slambook/ch7/1_depth.png \
  --depth1 data/slambook/ch7/2_depth.png \
  --matcher orb \
  --intrinsics 520.9 521.0 325.1 249.7 \
  --depth-scale 5000
```
