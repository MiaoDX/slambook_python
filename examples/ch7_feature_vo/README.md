# Chapter 7 Feature Visual Odometry

This directory contains command-line replacements for the legacy root scripts.

## 2D-2D Pose Estimation

```bash
python examples/ch7_feature_vo/pose_estimation_2d2d.py \
  --image0 data/slambook/ch7/1.png \
  --image1 data/slambook/ch7/2.png \
  --matcher orb
```

Use explicit intrinsics when they are known:

```bash
python examples/ch7_feature_vo/pose_estimation_2d2d.py \
  --image0 data/slambook/ch7/1.png \
  --image1 data/slambook/ch7/2.png \
  --matcher sift \
  --intrinsics 520.9 521.0 325.1 249.7
```

If no intrinsics are passed, the example uses a simple image-size-derived camera
matrix so the pipeline remains runnable for inspection. Pose estimates from that
fallback should not be treated as calibrated results.
