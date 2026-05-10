# Data Layout

Large datasets are not committed to this repository. Keep downloaded datasets
under `data/` locally, or pass absolute paths to examples through CLI flags.
See `docs/datasets.md` for the fuller dataset and trajectory-format notes.

Suggested layout:

```text
data/
  slambook/
    ch7/
      1.png
      2.png
      1_depth.png
    ch9/
      images/
        000000.png
        000001.png
  kitti/
    sequences/
    poses/
  tum_rgbd/
    rgb.txt
    depth.txt
    groundtruth.txt
    rgb/
    depth/
```

Examples should document the files they need and accept path arguments rather
than relying on this exact layout.
