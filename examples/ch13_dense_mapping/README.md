# Chapter 13 Dense Mapping

Fuse one RGB-D frame:

```bash
uv run python examples/ch13_dense_mapping/rgbd_fusion.py \
  --color data/slambook/ch13/color/1.png \
  --depth data/slambook/ch13/depth/1.pgm \
  --intrinsics FX FY CX CY \
  --depth-scale 5000 \
  --output outputs/ch13_frame.ply
```

Fuse a known-pose RGB-D sequence and export an occupancy grid:

```bash
uv run python examples/ch13_dense_mapping/rgbd_fusion.py \
  --color-dir data/slambook/ch13/color \
  --depth-dir data/slambook/ch13/depth \
  --pose-file data/slambook/ch13/pose.txt \
  --intrinsics FX FY CX CY \
  --depth-scale 5000 \
  --output outputs/ch13_map.ply \
  --occupancy-output outputs/ch13_occupancy.npz
```

Estimate a known-pose monocular depth map from a reference/current pair:

```bash
uv run python examples/ch13_dense_mapping/monocular_depth.py \
  --reference data/remode/images/000000.png \
  --current data/remode/images/000001.png \
  --transform-cur-ref 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 \
  --intrinsics FX FY CX CY \
  --output-depth outputs/ch13_mono_depth.npy
```
