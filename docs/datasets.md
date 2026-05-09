# Dataset Layout

Large datasets are not committed to this repository. Examples should accept
explicit path arguments and document any expected directory layout.

Suggested local layout:

```text
data/
  slambook/
    ch7/
      1.png
      2.png
      1_depth.png
  tum_rgbd/
    rgb.txt
    depth.txt
    groundtruth.txt
    rgb/
    depth/
  kitti/
    sequences/
    poses/
```

For RGB-D examples, depth scale must be explicit. TUM-style `uint16` depth often
uses `--depth-scale 5000`; metric floating-point depth should use
`--depth-scale 1`.

Trajectory outputs use:

- TUM RGB-D format: `timestamp tx ty tz qx qy qz qw`
- KITTI odometry format: one flattened `3x4` pose matrix per line
