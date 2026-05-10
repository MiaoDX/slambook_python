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

Chapter 9's mini VO runner expects one directory containing an ordered image
sequence. Filenames are sorted lexicographically, so zero-padded names are
recommended:

```text
data/
  slambook/
    ch9/
      images/
        000000.png
        000001.png
        000002.png
```

Run it with explicit intrinsics and optional trajectory outputs:

```bash
python examples/ch9_project/run_vo.py \
  --images data/slambook/ch9/images \
  --intrinsics FX FY CX CY \
  --output-tum outputs/ch9.tum \
  --output-kitti outputs/ch9.txt
```

For the depth-assisted local-map runner, provide a matching depth directory:

```bash
python examples/ch9_project/run_local_map_vo.py \
  --images data/slambook/ch9/images \
  --depths data/slambook/ch9/depth \
  --intrinsics FX FY CX CY \
  --depth-scale 5000 \
  --output-tum outputs/ch9_local_map.tum
```

For RGB-D examples, depth scale must be explicit. TUM-style `uint16` depth often
uses `--depth-scale 5000`; metric floating-point depth should use
`--depth-scale 1`.

The slambook `joinMap` / dense RGB-D examples use sorted `color/` and `depth/`
directories plus a `pose.txt` file with rows in this format:

```text
tx ty tz qx qy qz qw
```

Run the known-pose fusion path with:

```bash
python examples/ch13_dense_mapping/rgbd_fusion.py \
  --color-dir data/slambook/ch13/color \
  --depth-dir data/slambook/ch13/depth \
  --pose-file data/slambook/ch13/pose.txt \
  --intrinsics FX FY CX CY \
  --depth-scale 5000 \
  --output outputs/ch13_map.ply \
  --occupancy-output outputs/ch13_occupancy.npz
```

TUM RGB-D association files are parsed from `rgb.txt` and `depth.txt` using the
standard two-column format:

```text
timestamp relative/path.png
```

The helper associates RGB and depth frames by nearest timestamp within a
configurable tolerance.

Trajectory outputs use:

- TUM RGB-D format: `timestamp tx ty tz qx qy qz qw`
- KITTI odometry format: one flattened `3x4` pose matrix per line
