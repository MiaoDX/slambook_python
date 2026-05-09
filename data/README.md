# Data Layout

Large datasets are not committed to this repository. Keep downloaded datasets
under `data/` locally, or pass absolute paths to examples through CLI flags.

Suggested layout:

```text
data/
  slambook/
    ch7/
      1.png
      2.png
  kitti/
    sequences/
  tum_rgbd/
```

Examples should document the files they need and accept path arguments rather
than relying on this exact layout.
