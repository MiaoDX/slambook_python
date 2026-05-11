# Python SLAM Library Stack Research, 2026

Date: 2026-05-09

Scope: choose a practical Python stack for porting `gaoxiang12/slambook` into this repo while preserving the book's educational value.

## Executive Recommendation

Use a teaching-first Python stack with optional modern backends.

The port should not be a direct binding rewrite of Eigen, Sophus, Ceres, g2o, DBoW3, PCL, and Octomap. That would make the Python repo hard to install and harder to learn from. Instead:

- Keep the main examples runnable with NumPy, SciPy, OpenCV, matplotlib, and PyYAML.
- Use Open3D for point clouds, RGB-D examples, and dense mapping.
- Use SciPy `least_squares` for the first optimization chapters because it exposes residuals and Jacobians directly.
- Add GTSAM as the optional factor-graph backend for pose graphs and serious backend optimization.
- Add PyCOLMAP as an optional reference implementation for SfM, triangulation, bundle adjustment, and reconstruction data structures.
- Add Kornia / LightGlue / LoFTR as optional modern feature-matching variants, not as the baseline.
- Use FAISS for loop-closure retrieval experiments.
- Use Rerun and evo for visualization and trajectory evaluation.

Recommended dependency tiers:

```text
Core teaching stack:
  numpy
  scipy
  opencv-python
  matplotlib
  pyyaml
  tqdm

3D and evaluation stack:
  open3d
  evo
  rerun-sdk

Modern vision stack, optional:
  torch
  kornia
  lightglue
  pycolmap
  faiss-cpu

Backend optimization stack, optional:
  gtsam
  jax
  jaxlie
```

## Why This Split

The original slambook code is C++-native and uses the ecosystem that was natural for the book: Eigen for matrices, Sophus for Lie groups, OpenCV for vision, Ceres and g2o for optimization, DBoW3 for loop closure, and PCL / Octomap for dense mapping.

In Python, the high-quality path is different:

- NumPy and SciPy are the educational replacements for Eigen and hand-written linear algebra.
- OpenCV remains the right baseline for camera models, epipolar geometry, PnP, triangulation, and classic features.
- Open3D is a better Python-facing point-cloud and RGB-D mapping tool than PCL bindings.
- GTSAM has Python wrappers and maps naturally to factor graphs, pose graphs, and SLAM backends.
- PyCOLMAP exposes COLMAP's mature SfM machinery and gives us an industrial reference for features, tracks, reconstructions, triangulation, and bundle adjustment.
- Learned matchers are now useful, but they should be variants because they bring PyTorch and model-weight complexity.

## Current Repo Fit

This repo already partially covers slambook chapter 7:

- `pose_estimation_2d2d.py`: ORB features, BF matching, Fundamental Matrix, Essential Matrix, pose recovery.
- `pose_estimation_3d2d.py`: pixel-to-camera coordinates, triangulation, PnP.
- `simStereoCamera.py`: stereo rectification, disparity, RGB-D-style depth visualization.
- `ransac_test.py`: feature-match chunking and pose confidence experiments.
- `testRefine.py`: experiments around Fundamental Matrix and Homography inlier filtering.

The right first milestone is not adding more libraries. It is reorganizing the current scripts into importable modules, then adding modern backends behind stable interfaces.

Proposed package shape:

```text
slam/
  camera/
    pinhole.py
    stereo.py
  geometry/
    lie.py
    transforms.py
    triangulation.py
  features/
    opencv_features.py
    learned_features.py
    matching.py
  vo/
    two_view.py
    pnp.py
    visual_odometry.py
  optimization/
    scipy_ba.py
    gtsam_backend.py
  mapping/
    rgbd.py
    pointcloud.py
  io/
    datasets.py
    trajectory.py
  viz/
    matplotlib_viz.py
    open3d_viz.py
    rerun_viz.py
examples/
  ch3_geometry/
  ch4_lie/
  ch5_camera_image/
  ch6_optimization/
  ch7_feature_vo/
  ch8_direct_vo/
  ch9_project/
  ch10_bundle_adjustment/
  ch11_pose_graph/
  ch12_loop_closure/
  ch13_dense_mapping/
```

## Library Research By Subsystem

### 1. Core Linear Algebra And Geometry

Primary choices:

- `numpy`: base array and matrix representation.
- `scipy.spatial.transform`: rotations, quaternions, rotation vectors.
- `scipy.optimize`: nonlinear least squares and optimization.
- `spatialmath-python` or `pytransform3d`: optional human-friendly robotics transforms.
- `jaxlie`: optional if we later want JAX-native differentiable Lie groups.

Recommendation:

- Implement a tiny local `SO3` / `SE3` wrapper for the book examples.
- Use SciPy's `Rotation` internally for robust conversion between rotation matrices, quaternions, and rotation vectors.
- Keep `jaxlie` optional. It is excellent for differentiable SE(3) optimization, but JAX is too heavy for the baseline port.

Rationale:

- `jaxlie` directly targets SO(2), SE(2), SO(3), and SE(3), with exp/log, adjoints, inverse, composition, and tangent-space optimization helpers.
- `spatialmath-python` gives robotics-friendly SO/SE objects and notation.
- `pytransform3d` gives convention-heavy transform utilities and good visualization helpers.

Risk:

- Too many transform libraries cause convention drift. Pick one internal representation per module: 4x4 `T_wc` / `T_cw` matrices plus explicit docstrings.

### 2. Camera Models, Epipolar Geometry, PnP, Triangulation

Primary choice:

- `opencv-python`

Use for:

- camera calibration
- distortion models
- `findFundamentalMat`
- `findEssentialMat`
- `recoverPose`
- `solvePnP`
- `triangulatePoints`
- stereo calibration and rectification
- classic ORB/SIFT features

Recommendation:

- Keep OpenCV as required.
- Wrap OpenCV calls behind our own functions so examples stay readable and masks/inlier conventions are normalized.

Important implementation detail:

OpenCV masks are not always semantically identical. Some functions return 0/1 masks, while `recoverPose` masks can be 0/255 depending on binding/version. This is already noted in the current repo. Normalize masks with:

```python
inliers = mask.ravel() != 0
```

### 3. Feature Extraction And Matching

Baseline:

- OpenCV ORB + BFMatcher for free, fast, educational examples.
- OpenCV SIFT + FLANN for a stronger classic baseline.

Modern optional stack:

- `lightglue`: SuperPoint, DISK, ALIKED, SIFT, and DoGHardNet feature pipelines with LightGlue matching.
- `kornia`: PyTorch-native LoFTR and LightGlue interfaces.
- `hloc`: hierarchical localization toolbox using retrieval plus matching.
- `pycolmap`: SIFT and geometric verification through COLMAP's pipeline.

Recommendation:

- Baseline examples should stay OpenCV ORB/SIFT.
- Add a `FeatureMatcher` interface with implementations:
  - `ORBMatcher`
  - `SIFTMatcher`
  - `LightGlueMatcher` optional
  - `LoFTRMatcher` optional
  - `PyCOLMAPMatcher` optional, mainly for SfM/reference workflows

Why not require LightGlue:

- It brings PyTorch and model weights.
- It may need GPU for comfortable speed.
- It is better as a "modern variant" after the geometric baseline is understood.

Where it matters:

- low-texture image pairs
- wide-baseline matching
- poor repeatability from ORB
- visual localization experiments

### 4. Structure From Motion And Bundle Adjustment Reference

Primary optional choice:

- `pycolmap`

Use for:

- full SfM reference pipeline
- image database, camera models, tracks, reconstructions
- incremental mapping
- local and global bundle adjustment
- triangulation
- reading/writing COLMAP models

Recommendation:

- Do not base the educational implementation only on PyCOLMAP.
- Use PyCOLMAP as a reference backend and validation target:
  - compare our triangulation with COLMAP triangulation
  - compare our BA objective against PyCOLMAP/Ceres results
  - import/export reconstructions for visualization

Important 2026 note:

- Standalone GLOMAP is marked deprecated and migrated into COLMAP as the global mapper. Do not add a separate GLOMAP dependency for this repo. Use COLMAP/PyCOLMAP surfaces instead.

### 5. Nonlinear Least Squares And Bundle Adjustment

Teaching backend:

- `scipy.optimize.least_squares`

Serious backend:

- `gtsam`

Reference backend:

- `pycolmap`

Recommendation:

- Port ch6 curve fitting and ch10 toy bundle adjustment with SciPy first.
- Use explicit residual functions and sparse Jacobian structure where possible.
- Add GTSAM for pose graph and factor graph examples once the SciPy version is clear.
- Use PyCOLMAP when we want production-quality SfM/BA behavior or file-format interoperability.

Why SciPy first:

- It is easy to install.
- It teaches residual construction directly.
- It supports robust losses, bounds, sparse Jacobian structure, and nonlinear least squares.

Why GTSAM second:

- It is conceptually closer to modern SLAM backends.
- Factor graphs are the right abstraction for ch10/ch11.
- Python packaging can be less frictionless than SciPy, so it should not block the baseline.

Avoid initially:

- General Ceres/g2o Python bindings as required dependencies. They are either platform-sensitive or less standard than SciPy/GTSAM/PyCOLMAP.

### 6. Visual Odometry Project

The original `project/0.1` to `project/0.4` code defines the slambook mini visual odometry system:

- Camera
- Frame
- MapPoint
- Map
- VisualOdometry
- Config
- feature matching
- PnP pose estimation
- local map management
- bundle adjustment through g2o in later versions

Recommendation:

- Port this as real Python package code, not as notebooks or standalone scripts.
- Keep data classes small and explicit:
  - `Camera`
  - `Frame`
  - `MapPoint`
  - `Map`
  - `VisualOdometry`
  - `VisualOdometryConfig`
- Keep the first VO version OpenCV-only.
- Add optional local bundle adjustment after the pipeline is stable.

Possible backend levels:

```text
Level 0: OpenCV PnP only
Level 1: OpenCV PnP + local map + keyframes
Level 2: SciPy motion-only BA
Level 3: GTSAM factor graph backend
Level 4: PyCOLMAP reference comparison
```

### 7. Direct Visual Odometry

Relevant original chapter:

- ch8 direct sparse and semi-dense methods

Required primitives:

- image pyramids
- interpolation
- photometric residuals
- robust losses
- SE(3) updates
- Jacobians

Recommendation:

- Implement the educational direct method with NumPy/SciPy/OpenCV first.
- Use Numba only if performance blocks comprehension.
- Avoid a deep-learning SLAM dependency here; it changes the lesson.

Optional research references:

- DROID-SLAM is a strong learned SLAM system for monocular, stereo, and RGB-D, but it requires CUDA/GPU and is not a drop-in replacement for a teaching port.

### 8. Point Clouds, RGB-D, Dense Reconstruction

Primary choice:

- `open3d`

Use for:

- point cloud creation from RGB-D
- voxel downsampling
- normal estimation
- ICP
- RGB-D odometry
- TSDF-style integration
- mesh reconstruction
- visualization
- reading/writing PLY/PCD

Recommendation:

- Use Open3D as the required 3D dependency once we start ch5 joinMap and ch13 dense mapping.
- Replace PCL/Octomap examples with Open3D equivalents where educationally acceptable.
- Keep Octomap as "not ported directly" unless there is a specific occupancy-grid requirement.

### 9. Loop Closure And Image Retrieval

Original dependency:

- DBoW3

Python alternatives:

- FAISS for vector similarity search
- hloc for retrieval-plus-matching workflows
- global descriptor models, optional

Recommendation:

- Implement a simple bag-of-visual-words baseline for teaching if needed.
- Use FAISS as the modern retrieval index.
- Use hloc as a reference for image retrieval and feature matching pipelines.

Key boundary:

- FAISS is an index/search library, not a descriptor extractor. We still need descriptors from SIFT, learned global descriptors, NetVLAD-style models, or another image embedding source.

### 10. Visualization And Debugging

Choices:

- matplotlib for static plots
- OpenCV windows for quick image debugging
- Open3D for 3D geometry visualization
- Rerun for time-series robotics/computer-vision logging

Recommendation:

- Use matplotlib and Open3D in examples.
- Add Rerun as optional but strongly recommended for VO and SLAM debugging.

Good Rerun use cases:

- camera images over time
- feature tracks
- point clouds
- trajectory poses
- per-frame inlier counts
- reprojection errors

### 11. Trajectory Evaluation

Primary choice:

- `evo`

Use for:

- ATE and RPE metrics
- comparing estimated trajectories to ground truth
- KITTI/TUM-style trajectory formats
- visualizing trajectory alignment

Recommendation:

- Standardize this repo's output trajectory format early.
- Prefer TUM format for timestamped trajectories and KITTI format for KITTI odometry examples.
- Add a small `slam.io.trajectory` module for export.

## Chapter-To-Library Mapping

| Slambook area | Python baseline | Optional modern/reference |
| --- | --- | --- |
| ch2 build basics | plain Python scripts | none |
| ch3 rigid body motion | NumPy, SciPy Rotation | pytransform3d, spatialmath-python |
| ch4 Lie groups | local SO3/SE3 module | jaxlie, spatialmath-python |
| ch5 camera/images | OpenCV, NumPy | Open3D for RGB-D point clouds |
| ch6 nonlinear optimization | SciPy least_squares | GTSAM for factor examples |
| ch7 feature VO | OpenCV ORB/SIFT + calib3d | LightGlue, Kornia, PyCOLMAP |
| ch8 direct VO | OpenCV, NumPy, SciPy | JAX/JAXLie for differentiable variants |
| ch9 VO project | local Python package | GTSAM/PyCOLMAP backends |
| ch10 bundle adjustment | SciPy sparse BA | GTSAM, PyCOLMAP |
| ch11 pose graph | SciPy small examples | GTSAM |
| ch12 loop closure | simple BoW, OpenCV descriptors | FAISS, hloc |
| ch13 dense mapping | Open3D | Rerun for debugging |

## Recommended Interfaces

Do not let individual examples call every library directly. Create small interfaces that preserve the educational vocabulary.

### Feature Matching

```python
class MatchResult:
    keypoints0: np.ndarray
    keypoints1: np.ndarray
    matches: np.ndarray
    scores: np.ndarray | None

class FeatureMatcher:
    def match(self, image0: np.ndarray, image1: np.ndarray) -> MatchResult:
        ...
```

Implementations:

- `ORBMatcher`
- `SIFTMatcher`
- `LightGlueMatcher`
- `LoFTRMatcher`

### Pose Estimation

```python
class TwoViewPose:
    R_10: np.ndarray
    t_10: np.ndarray
    inliers: np.ndarray
    E: np.ndarray | None
    F: np.ndarray | None
```

Key convention:

- Document whether `R_10, t_10` maps camera 0 into camera 1 or the inverse.
- Use one convention across all examples.

### Optimization

```python
class BundleAdjustmentBackend:
    def optimize(self, problem: BAProblem) -> BAResult:
        ...
```

Implementations:

- `ScipyBABackend`
- `GtsamBABackend`
- `PyColmapReferenceBackend`

### Visualization

```python
class SlamLogger:
    def log_image(self, name, image, t): ...
    def log_matches(self, name, image0, image1, matches, t): ...
    def log_pose(self, name, T_wc, t): ...
    def log_points(self, name, points, colors=None, t=None): ...
```

Implementations:

- `MatplotlibLogger`
- `Open3DLogger`
- `RerunLogger`

## Migration Plan

### Phase 1: Clean Chapter 7 Baseline

Goal: turn existing scripts into reusable OpenCV baseline modules.

Work:

- Create `slam.features.opencv_features`.
- Create `slam.geometry.triangulation`.
- Create `slam.vo.two_view`.
- Normalize masks and coordinate conventions.
- Move demo code into `examples/ch7_feature_vo/`.
- Add tests with synthetic camera poses and synthetic point clouds.

Dependencies:

- NumPy
- SciPy
- OpenCV
- matplotlib

### Phase 2: Camera And Geometry Foundations

Goal: cover ch3, ch4, ch5 basics.

Work:

- Add `Camera` and `StereoCamera`.
- Add `SE3` helper functions.
- Add coordinate-frame docs.
- Add RGB-D to point cloud example with Open3D.

Dependencies:

- Open3D starts here.

### Phase 3: Optimization

Goal: cover ch6 and initial ch10.

Work:

- Curve fitting with SciPy.
- Pose-only optimization.
- Small bundle adjustment.
- Sparse Jacobian example.

Dependencies:

- SciPy only for baseline.

### Phase 4: Mini VO Project

Goal: port `project/0.1` through `project/0.4`.

Work:

- Implement `Camera`, `Frame`, `MapPoint`, `Map`, `VisualOdometry`.
- Add keyframe insertion.
- Add local map matching.
- Add PnP pose estimation.
- Add optional SciPy/GTSAM refinement.
- Export trajectory in TUM/KITTI format.

Dependencies:

- OpenCV baseline.
- evo for evaluation.
- Rerun optional for debugging.

### Phase 5: Modern Variants

Goal: add modern feature and SfM references without destabilizing the baseline.

Work:

- Add LightGlue matcher.
- Add LoFTR matcher.
- Add PyCOLMAP reference examples.
- Add hloc-style retrieval/matching example if useful.

Dependencies:

- PyTorch, Kornia, LightGlue, PyCOLMAP optional.

### Phase 6: Backend, Loop Closure, Dense Mapping

Goal: cover ch11, ch12, ch13.

Work:

- GTSAM pose graph.
- FAISS retrieval for loop closure.
- Open3D dense RGB-D mapping.
- Rerun visualization of SLAM state.

## Key Risks

### Risk: Dependency sprawl

Mitigation:

- Use optional extras:

```text
uv sync --extra core --frozen
uv sync --extra 3d --frozen
uv sync --extra modern --frozen
uv sync --extra backend --frozen
uv sync --all-extras --frozen
```

### Risk: Coordinate-frame confusion

Mitigation:

- Define conventions in `docs/coordinates.md`.
- Name transforms explicitly: `T_wc`, `T_cw`, `T_10`, `T_01`.
- Include unit tests for composition and inverse.

### Risk: Learned matchers hide geometry

Mitigation:

- Keep ORB/SIFT examples first.
- Make modern matchers a swap-in backend after the geometric pipeline is visible.

### Risk: PyCOLMAP becomes a black box

Mitigation:

- Treat it as a reference and interop layer.
- Keep hand-written educational implementations for two-view pose, triangulation, and small BA.

### Risk: GTSAM packaging friction

Mitigation:

- Keep GTSAM optional.
- Provide SciPy equivalents for small examples.

## Decision Matrix

| Library | Role | Required? | Why | Main risk |
| --- | --- | --- | --- | --- |
| NumPy | arrays, matrices | yes | universal Python numeric base | none |
| SciPy | optimization, rotations | yes | strong teaching fit, easy install | slower than specialized C++ BA |
| OpenCV | vision geometry | yes | still the best Python baseline for calib3d | mask/version quirks |
| matplotlib | plots | yes | simple educational visualization | weak for live SLAM |
| PyYAML | configs | yes | VO project configs | none |
| Open3D | point clouds, RGB-D, dense mapping | yes after ch5/ch13 | best Python-facing 3D toolkit | heavy wheel |
| evo | trajectory evaluation | recommended | standard VO/SLAM metrics | GPL license consideration |
| Rerun | live debugging | optional recommended | excellent robotics/CV logging | extra viewer dependency |
| LightGlue | modern sparse matching | optional | strong wide-baseline matcher | PyTorch/model weights/GPU |
| Kornia | LoFTR, PyTorch vision geometry | optional | convenient learned matching APIs | PyTorch dependency |
| PyCOLMAP | SfM/BA reference | optional recommended | mature COLMAP in Python | can become black box |
| GTSAM | factor graph backend | optional recommended | right abstraction for SLAM backend | install/platform friction |
| FAISS | image retrieval index | optional | fast similarity search | needs descriptor source |
| jaxlie | differentiable Lie groups | optional | strong for JAX optimization | JAX stack is heavy |
| spatialmath-python | robotics transforms | optional | readable SE3/SO3 API | another convention source |
| pytransform3d | transform utilities | optional | good conventions and visualization | another convention source |

## Concrete Recommendation For This Repo

Start with this dependency set:

```text
numpy
scipy
opencv-python
matplotlib
pyyaml
tqdm
open3d
evo
```

Do not install the modern stack until the baseline package is clean:

```text
torch
kornia
lightglue
pycolmap
faiss-cpu
rerun-sdk
gtsam
jax
jaxlie
```

First implementation milestone:

```text
Port and clean ch7 as a package:
  existing pose_estimation_2d2d.py -> slam/features + slam/vo/two_view.py
  existing pose_estimation_3d2d.py -> slam/geometry/triangulation.py + slam/vo/pnp.py
  existing simStereoCamera.py -> slam/camera/stereo.py + examples/ch5_or_ch13
```

Definition of done:

- `examples/ch7_feature_vo/pose_estimation_2d2d.py` runs from CLI.
- It writes a small result report: number of matches, inliers, estimated `R`, estimated `t`.
- It can switch `--matcher orb` and `--matcher sift`.
- Unit tests cover triangulation, mask normalization, and a synthetic two-view pose.
- No learned matcher dependency is required.

Second milestone:

- Add `--matcher lightglue` behind an optional extra.
- Compare ORB/SIFT/LightGlue on the same image pair.
- Keep the same downstream `findEssentialMat` / `recoverPose` pipeline.

Third milestone:

- Add Open3D RGB-D point cloud and trajectory visualization.
- Add evo trajectory export.

## Sources Checked

- Original upstream repo and chapter layout: https://github.com/gaoxiang12/slambook
- OpenCV calib3d documentation: https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html
- SciPy `least_squares`: https://scipy.github.io/devdocs/reference/generated/scipy.optimize.least_squares.html
- GTSAM docs: https://gtsam.org/docs/
- PyCOLMAP docs: https://colmap.github.io/pycolmap/pycolmap.html
- GLOMAP repo status and COLMAP global mapper note: https://github.com/colmap/glomap
- Open3D docs: https://open3d.org/html/
- Open3D RGB-D odometry docs: https://www.open3d.org/docs/release/tutorial/pipelines/rgbd_odometry.html
- LightGlue repo: https://github.com/cvg/LightGlue
- Kornia feature docs: https://kornia.readthedocs.io/en/latest/feature.html
- hloc repo: https://github.com/cvg/Hierarchical-Localization
- jaxlie docs: https://brentyi.github.io/jaxlie/
- spatialmath-python docs: https://spatialmath-python.rai-inst.com/
- pytransform3d docs: https://dfki-ric.github.io/pytransform3d/
- FAISS docs: https://faiss.ai/
- Rerun repo: https://github.com/rerun-io/rerun
- evo package/docs entry: https://pypi.org/project/evo/
