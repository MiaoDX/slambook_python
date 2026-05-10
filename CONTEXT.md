# Slambook Python Context

## Project Goal

This repository is a teaching-first Python port of selected `slambook` examples.
The maintained implementation lives in `slam/` and `examples/`; historical root
scripts live in `legacy/`.

## Glossary

- **Baseline backend**: the dependency-light implementation used for teaching
  and core tests. Baseline code uses NumPy, SciPy, OpenCV, Matplotlib, PyYAML,
  and small local helpers.
- **Optional backend**: a modern or platform-sensitive backend behind an extra,
  such as GTSAM, PyCOLMAP, LightGlue, LoFTR, FAISS, Open3D, Rerun, JAX, or
  JAXLie.
- **Camera intrinsics**: pinhole parameters `fx`, `fy`, `cx`, `cy`, represented
  by `CameraIntrinsics`.
- **Distortion coefficients**: Brown-Conrady coefficients in OpenCV order
  `[k1, k2, p1, p2, k3]`.
- **Transform `T_ab`**: homogeneous transform that maps coordinates from frame
  `b` into frame `a`.
- **World pose `T_wc`**: camera pose in world coordinates.
- **Camera pose `T_cw`**: inverse of `T_wc`; maps world coordinates into the
  camera.
- **Relative pose `T_10`**: transform from camera 0 coordinates into camera 1
  coordinates, following OpenCV `recoverPose`.
- **Frame**: one image plus timestamp, pose, keypoints, and descriptors.
- **Keyframe**: a frame retained in the map for future local-map matching.
- **MapPoint**: a 3D landmark in world coordinates with descriptor and
  observations.
- **Local map**: the current keyframes and map points used to track a new frame.
- **Motion-only BA**: pose-only reprojection refinement with fixed 3D map points.
- **BAL problem**: bundle-adjustment input with cameras, 3D points, and 2D
  observations.
- **Pose graph**: vertices as poses and edges as relative pose constraints.
- **BoW descriptor**: bag-of-visual-words histogram produced from local OpenCV
  descriptors and a trained `VisualVocabulary`.
- **Occupancy grid**: occupied voxel centers and counts derived from fused point
  clouds, used as the Python-native Octomap-style artifact.
- **Validation smoke test**: command run against representative sample data to
  prove an example executes and produces inspectable numeric output.
- **Benchmark report**: structured metrics comparing an estimate against a
  reference, such as trajectory RMSE or reprojection RMSE.

## Source Of Truth

- Chapter coverage: `docs/status.md`
- Dataset layout: `docs/datasets.md`
- Coordinate conventions: `docs/coordinates.md`
- Porting rules: `docs/porting.md`
- Validation evidence: `docs/validation.md`
- Legacy scripts: `legacy/README.md`
