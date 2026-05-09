# Slambook Python Migration Plan

Date: 2026-05-09

Source research: `docs/research/python-slam-library-stack-2026.md`

Objective: migrate `gaoxiang12/slambook` into a maintainable, teaching-first Python project while preserving the book's core SLAM concepts and adding optional modern Python backends where they improve comparison, debugging, or practical use.

## Migration Principles

1. Keep the baseline educational and easy to install.
2. Port concepts, examples, and experiments chapter by chapter; do not mechanically translate C++ syntax.
3. Use stable internal interfaces so OpenCV, SciPy, GTSAM, PyCOLMAP, LightGlue, and Open3D can be swapped without rewriting examples.
4. Separate required dependencies from optional modern backends.
5. Add tests around geometry, masks, coordinate-frame conventions, and optimization residuals before expanding feature surface.
6. Keep examples executable from the command line and make each one produce inspectable output.

## Target Dependency Policy

Core dependencies:

```text
numpy
scipy
opencv-python
matplotlib
pyyaml
tqdm
```

3D and evaluation dependencies:

```text
open3d
evo
```

Optional modern dependencies:

```text
rerun-sdk
torch
kornia
lightglue
pycolmap
faiss-cpu
gtsam
jax
jaxlie
```

Packaging target:

```text
pip install -e .[core]
pip install -e .[3d]
pip install -e .[modern]
pip install -e .[backend]
pip install -e .[all]
```

## Target Repository Shape

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
    base.py
    opencv_features.py
    learned_features.py
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
tests/
  geometry/
  features/
  vo/
  optimization/
```

## Milestone 0: Project Scaffolding

Goal: make the repo installable and give future chapter ports a stable home.

Inputs:

- Existing root scripts.
- Existing `README.md`.
- Existing agent docs.

Work:

- Add `pyproject.toml` with package metadata and optional extras.
- Add `slam/` and `examples/` package directories.
- Add `tests/` and a minimal `pytest` setup.
- Add `docs/coordinates.md` defining coordinate-frame conventions.
- Add `docs/porting.md` with rules for translating slambook C++ examples into Python.
- Add `data/README.md` explaining how datasets are expected to be arranged without committing large datasets.
- Keep current root scripts untouched until their replacements are working.

Deliverables:

- `pip install -e .[core]` works.
- `pytest` runs and passes at least smoke tests.
- Documentation defines transform names such as `T_wc`, `T_cw`, `T_10`, and `T_01`.

Acceptance criteria:

- No optional backend is required to import `slam`.
- Fresh checkout can run `python -m pytest`.
- The root scripts still exist and still represent the legacy baseline.

Risk:

- Premature packaging churn.

Mitigation:

- Keep setup minimal. Do not introduce build tools beyond standard Python packaging unless needed.

## Milestone 1: Clean Chapter 7 Baseline

Goal: turn the current feature-based visual odometry scripts into reusable OpenCV-based modules.

Source files:

- `pose_estimation_2d2d.py`
- `pose_estimation_3d2d.py`
- `utils.py`
- `ransac_test.py`
- `testRefine.py`

Work:

- Extract feature detection and matching into `slam.features`.
- Extract two-view geometry into `slam.vo.two_view`.
- Extract triangulation into `slam.geometry.triangulation`.
- Extract PnP helpers into `slam.vo.pnp`.
- Normalize all OpenCV masks with `mask.ravel() != 0`.
- Replace global `DEBUG` behavior with explicit CLI flags.
- Add CLI examples:
  - `examples/ch7_feature_vo/pose_estimation_2d2d.py`
  - `examples/ch7_feature_vo/pose_estimation_3d2d.py`
  - `examples/ch7_feature_vo/triangulation.py`
- Add matcher switch:
  - `--matcher orb`
  - `--matcher sift`
- Add image path arguments instead of hard-coded Windows paths.
- Add structured result printing:
  - match count
  - Fundamental Matrix inlier count
  - Essential Matrix inlier count
  - recovered `R`
  - recovered `t`
  - triangulated point count

Tests:

- Synthetic two-camera scene recovers a plausible relative pose.
- Triangulation returns finite 3D points for known projections.
- Mask normalization treats 0/1 and 0/255 masks equivalently.
- PnP smoke test recovers a pose from synthetic 3D/2D correspondences.

Deliverables:

- Clean importable modules for the current repo's strongest area.
- CLI examples replacing the old script workflows.
- Legacy root scripts either remain as compatibility wrappers or are documented as deprecated.

Acceptance criteria:

- `python examples/ch7_feature_vo/pose_estimation_2d2d.py --image0 ... --image1 ... --matcher orb` runs.
- The same CLI runs with `--matcher sift` where OpenCV SIFT is available.
- Tests pass without GPU or optional modern dependencies.

Risk:

- Existing code has implicit shape conventions like `Nx2`, `Nx3`, `3xN`.

Mitigation:

- Encode shapes in docstrings and tests. Prefer `Nx2` and `Nx3` arrays at public boundaries.

## Milestone 2: Chapter 3-5 Geometry, Camera, And RGB-D Foundations

Goal: add the mathematical foundation chapters and clean camera abstractions needed by later VO work.

Source areas:

- Upstream `ch3`
- Upstream `ch4`
- Upstream `ch5`
- Existing `simStereoCamera.py`

Work:

- Implement `slam.geometry.transforms`:
  - homogeneous transform construction
  - inverse
  - composition
  - point transformation
  - rotation-vector conversion through SciPy
- Implement `slam.geometry.lie`:
  - minimal `SO3` / `SE3` exp and log
  - perturbation helpers for optimization examples
- Implement `slam.camera.pinhole`:
  - `CameraIntrinsics`
  - pixel-to-camera coordinates
  - camera-to-pixel projection
  - distortion placeholder API
- Implement `slam.camera.stereo`:
  - rectification wrapper
  - disparity-to-depth helper
  - stereo baseline conventions
- Port `ch5/imageBasics` as an OpenCV image example.
- Port `ch5/joinMap` with Open3D point cloud generation.
- Move relevant parts of `simStereoCamera.py` into `slam.camera.stereo` and examples.

Tests:

- Transform inverse and composition identity checks.
- Pixel/camera projection round trip with known intrinsics.
- `SE3.exp(SE3.log(T))` round trip for small motions.
- Stereo disparity-to-depth with known focal length and baseline.

Deliverables:

- Geometry and camera modules used by later examples.
- Open3D RGB-D point cloud example.

Acceptance criteria:

- ch3/ch4/ch5 examples run with core plus `open3d` for RGB-D point cloud examples.
- Coordinate convention docs match actual transform behavior in tests.

Risk:

- Lie group implementation can become a rabbit hole.

Mitigation:

- Keep it minimal. Use SciPy for rotation conversions and implement only the formulas needed by the book examples.

## Milestone 3: Chapter 6 Optimization

Goal: port nonlinear optimization examples in a way that teaches residual construction before introducing SLAM backends.

Source areas:

- Upstream `ch6/ceres_curve_fitting`
- Upstream `ch6/g2o_curve_fitting`

Work:

- Implement curve fitting with `scipy.optimize.least_squares`.
- Add robust loss examples.
- Add explicit residual functions and optional analytic Jacobians.
- Document mapping from Ceres/g2o concepts to SciPy concepts:
  - residual block -> residual function slice
  - parameter block -> vector segment
  - robust kernel -> `loss`
  - solver options -> `least_squares` options

Tests:

- Synthetic noisy curve fitting recovers parameters within tolerance.
- Robust loss is less sensitive to injected outliers than linear loss.

Deliverables:

- `examples/ch6_optimization/curve_fitting.py`
- `slam.optimization` utilities for residual packing where useful.

Acceptance criteria:

- Example prints initial/final parameter error and final cost.
- No Ceres/g2o dependency is required.

Risk:

- The port could hide the optimization lesson behind helpers.

Mitigation:

- Keep the first example explicit even if it repeats some code.

## Milestone 4: Chapter 9 Mini Visual Odometry Project

Goal: port the `project/0.1` through `project/0.4` mini VO system into a real Python package.

Source areas:

- Upstream `project/0.1`
- Upstream `project/0.2`
- Upstream `project/0.3`
- Upstream `project/0.4`

Core classes:

- `Camera`
- `Frame`
- `MapPoint`
- `Map`
- `VisualOdometry`
- `VisualOdometryConfig`

Work:

- Implement the data model with dataclasses where appropriate.
- Implement frame creation and feature extraction.
- Implement PnP pose estimation.
- Implement map point insertion and tracking.
- Implement keyframe insertion.
- Implement local map matching.
- Add config loading through PyYAML.
- Add trajectory export in TUM and KITTI formats.
- Add an example runner under `examples/ch9_project/run_vo.py`.

Backend levels:

```text
Level 0: OpenCV PnP only
Level 1: OpenCV PnP + local map + keyframes
Level 2: SciPy motion-only BA
Level 3: optional GTSAM backend
Level 4: optional PyCOLMAP reference comparison
```

Tests:

- `Frame` and `MapPoint` lifecycle tests.
- Camera projection tests reused from Milestone 2.
- VO smoke test on a tiny synthetic or fixture sequence.
- Trajectory export format tests.

Deliverables:

- A usable Python VO mini-project.
- Trajectory files compatible with evo.

Acceptance criteria:

- `examples/ch9_project/run_vo.py --config ...` runs on a documented dataset layout.
- Estimated poses are exported.
- The runner does not require GTSAM, PyCOLMAP, LightGlue, or GPU.

Risk:

- Dataset availability blocks testing.

Mitigation:

- Add synthetic fixtures and document real dataset download/layout separately.

## Milestone 5: Chapter 10 Bundle Adjustment

Goal: port bundle adjustment examples with both educational and reference backends.

Source areas:

- Upstream `ch10/ceres_custombundle`
- Upstream `ch10/g2o_custombundle`

Work:

- Implement BAL data reader in `slam.io.datasets`.
- Implement camera and point parameter packing.
- Implement reprojection residuals.
- Implement SciPy sparse bundle adjustment.
- Add optional PyCOLMAP reference comparison.
- Add optional GTSAM backend only after SciPy version is correct.

Tests:

- BAL parser reads camera, point, and observation counts correctly.
- Residual vector shape is correct.
- Bundle adjustment reduces reprojection RMSE on a small problem.

Deliverables:

- `examples/ch10_bundle_adjustment/scipy_bal.py`
- Optional `examples/ch10_bundle_adjustment/pycolmap_reference.py`

Acceptance criteria:

- Small BAL problem runs with SciPy.
- Report includes initial and final reprojection error.

Risk:

- Large BA can be slow in pure SciPy.

Mitigation:

- Start with small BAL fixtures. Use sparse Jacobian structure and make large problems optional.

## Milestone 6: Chapter 11 Pose Graph

Goal: port pose graph optimization using a clear baseline and a proper factor-graph backend.

Source areas:

- Upstream `ch11`
- Upstream `pose_graph`

Work:

- Read `.g2o` pose graph files.
- Implement small SciPy pose graph example for teaching.
- Implement GTSAM pose graph optimizer as the practical backend.
- Export optimized trajectory.
- Add plotting through matplotlib and optional Rerun.

Tests:

- `.g2o` parser smoke test.
- Known small graph optimization reduces edge error.
- GTSAM backend is skipped cleanly when not installed.

Deliverables:

- `slam.optimization.pose_graph`
- `examples/ch11_pose_graph/optimize_pose_graph.py`

Acceptance criteria:

- Example can optimize a small included `.g2o` fixture.
- Optional GTSAM path gives comparable or better residual than baseline.

Risk:

- GTSAM installation can fail on some platforms.

Mitigation:

- Keep GTSAM optional and skip tests when missing.

## Milestone 7: Chapter 8 Direct Visual Odometry

Goal: port direct sparse/semi-dense visual odometry after the transform and optimization foundations are stable.

Source areas:

- Upstream `ch8/LKFlow`
- Upstream `ch8/directMethod`

Work:

- Implement LK optical flow example with OpenCV.
- Implement image pyramid utilities.
- Implement bilinear interpolation.
- Implement photometric residuals.
- Implement direct pose refinement with SciPy.
- Keep JAX/JAXLie as optional future work, not part of the initial port.

Tests:

- Interpolation test on known image values.
- Photometric residual shape and sign tests.
- Synthetic image warp recovers a small pose update within tolerance.

Deliverables:

- `examples/ch8_direct_vo/lk_flow.py`
- `examples/ch8_direct_vo/direct_sparse.py`

Acceptance criteria:

- Examples run with core dependencies.
- Direct method documents its limits and expected input assumptions.

Risk:

- Direct methods are sensitive to photometric assumptions.

Mitigation:

- Use small controlled examples first, then real sequences.

## Milestone 8: Chapter 12 Loop Closure

Goal: port loop closure concepts with both a simple educational baseline and a modern retrieval index.

Source areas:

- Upstream `ch12`

Work:

- Implement simple image retrieval using OpenCV descriptors.
- Add FAISS-backed vector search.
- Add optional hloc-style retrieval/matching comparison.
- Keep DBoW3 as a conceptual source, not a required Python dependency.

Tests:

- FAISS retrieval returns expected nearest vectors on synthetic data.
- Loop candidate filtering avoids immediate temporal neighbors.

Deliverables:

- `slam.io.image_retrieval`
- `examples/ch12_loop_closure/retrieve_candidates.py`

Acceptance criteria:

- Baseline retrieval runs without FAISS.
- FAISS path is enabled behind an optional extra.

Risk:

- FAISS does not generate descriptors.

Mitigation:

- Make descriptor source explicit: SIFT, simple BoW, or learned global descriptor.

## Milestone 9: Chapter 13 Dense Mapping

Goal: port RGB-D dense reconstruction using Open3D as the Python-native replacement for PCL/Octomap-style workflows.

Source areas:

- Upstream `ch13/dense_RGBD`
- Upstream `ch13/dense_monocular`

Work:

- Implement RGB-D frame loading.
- Create Open3D point clouds from color/depth pairs.
- Fuse multiple RGB-D frames with known poses.
- Add voxel downsampling and normal estimation.
- Add optional mesh reconstruction.
- Add optional Rerun logging.

Tests:

- RGB-D to point cloud returns expected point counts on tiny fixtures.
- Pose transform moves point clouds consistently.

Deliverables:

- `slam.mapping.rgbd`
- `slam.mapping.pointcloud`
- `examples/ch13_dense_mapping/rgbd_fusion.py`

Acceptance criteria:

- Example writes a point cloud file.
- Visualization works with Open3D.

Risk:

- Original Octomap occupancy behavior is not reproduced exactly.

Mitigation:

- Document this as an Open3D-based equivalent unless exact occupancy mapping becomes a requirement.

## Milestone 10: Modern Matching And Reference Backends

Goal: add modern feature matching and SfM reference paths only after the baseline port is stable.

Work:

- Add `LightGlueMatcher`.
- Add `LoFTRMatcher` through Kornia or a dedicated wrapper.
- Add `PyCOLMAPMatcher` or PyCOLMAP reconstruction reference examples.
- Add Rerun logging for matches, tracks, poses, and point clouds.
- Compare OpenCV ORB/SIFT against LightGlue/LoFTR on the same examples.

Tests:

- Optional dependency tests are marked and skipped when packages are missing.
- Match result interface remains identical across matcher implementations.

Deliverables:

- `slam.features.learned_features`
- `examples/ch7_feature_vo/compare_matchers.py`
- `examples/reference/pycolmap_reconstruction.py`

Acceptance criteria:

- Core package still imports without modern dependencies.
- Modern examples fail with clear install guidance if optional dependencies are missing.

Risk:

- Learned matchers can dominate the repo and obscure the book's geometry lessons.

Mitigation:

- Keep them under `modern` extras and comparison examples.

## Cross-Cutting Workstreams

### Documentation

- Update `README.md` with project goal, install extras, and chapter status.
- Add `docs/status.md` tracking each chapter's port status.
- Add `docs/coordinates.md`.
- Add `docs/datasets.md`.
- Add per-example README files for nontrivial chapters.

### Testing

- Use synthetic tests before real datasets.
- Add optional dependency markers:
  - `pytest.mark.open3d`
  - `pytest.mark.gtsam`
  - `pytest.mark.pycolmap`
  - `pytest.mark.modern`
- Keep CI core-only at first.

### Data Policy

- Do not commit large datasets.
- Commit tiny generated fixtures only.
- Provide scripts or docs for expected dataset placement.
- Prefer CLI args over hard-coded paths.

### Backwards Compatibility

- Keep root scripts during the first migration wave.
- After example replacements are stable, either:
  - convert root scripts into wrappers, or
  - move them into `legacy/` with documentation.

### Quality Gates

Every milestone should include:

- runnable CLI example
- test coverage for math or data parsing logic
- documented dependencies
- no required optional modern backends
- no hard-coded local paths

## Initial Task Breakdown

Recommended first implementation sequence:

1. Add package scaffold and `pyproject.toml`.
2. Add `docs/coordinates.md` and decide transform naming.
3. Add `slam.features.base` and `slam.features.opencv_features`.
4. Move ORB matching from `pose_estimation_2d2d.py` into `ORBMatcher`.
5. Add SIFT matcher.
6. Add `slam.vo.two_view` with `estimate_fundamental`, `estimate_essential`, and `recover_pose`.
7. Add mask normalization utility.
8. Add synthetic two-view tests.
9. Add `examples/ch7_feature_vo/pose_estimation_2d2d.py`.
10. Add README status table.

This gives the repo a stable pattern before touching optimization, project VO, or modern backends.

## Chapter Status Tracking Template

| Chapter | Topic | Status | Baseline backend | Optional backend | Notes |
| --- | --- | --- | --- | --- | --- |
| ch2 | Python/project basics | Not started | Python packaging | none | Replace CMake lesson with Python packaging notes |
| ch3 | Rigid body motion | Not started | NumPy/SciPy | pytransform3d | Needed by all later chapters |
| ch4 | Lie groups | Not started | local SO3/SE3 | jaxlie/spatialmath | Keep minimal |
| ch5 | Camera and images | Not started | OpenCV | Open3D | Current stereo script overlaps |
| ch6 | Nonlinear optimization | Not started | SciPy | GTSAM | Curve fitting first |
| ch7 | Feature VO | Partially legacy | OpenCV | LightGlue/PyCOLMAP | Current repo's strongest area |
| ch8 | Direct VO | Not started | OpenCV/SciPy | JAX | Do after ch6 |
| ch9 | VO project | Not started | OpenCV | GTSAM/PyCOLMAP | Main package milestone |
| ch10 | Bundle adjustment | Not started | SciPy | GTSAM/PyCOLMAP | Needs BAL parser |
| ch11 | Pose graph | Not started | SciPy small graph | GTSAM | Needs g2o parser |
| ch12 | Loop closure | Not started | OpenCV descriptors | FAISS/hloc | Descriptor source explicit |
| ch13 | Dense mapping | Not started | Open3D | Rerun | Replace PCL/Octomap concepts |

## Commit Strategy

Use small commits by milestone slice:

1. `Add Python package scaffold`
2. `Document coordinate conventions`
3. `Add OpenCV feature matcher interfaces`
4. `Add two-view pose estimation module`
5. `Port ch7 2d2d example`
6. `Add synthetic geometry tests`

Avoid one large "port slambook" commit. Each commit should leave examples and tests runnable.

## Open Decisions

1. Python version floor: recommend Python 3.10+ because evo currently requires Python 3.10+.
2. Package manager: standard `pip` + `pyproject.toml` is enough at first.
3. Dataset priority: choose between using slambook sample data first or adding KITTI/TUM support early.
4. License note: upstream slambook is MIT; evo is GPLv3, so keep evo usage as an optional tool unless GPL implications are acceptable for downstream distribution.
5. Exact coordinate convention: decide and document before porting VO internals.

## Next Recommended Action

Start Milestone 0 and the first half of Milestone 1:

- Add `pyproject.toml`.
- Add package directories.
- Add coordinate docs.
- Extract current ORB/SIFT matching into `slam.features`.
- Add two-view pose estimation helpers.
- Add one CLI example for ch7 2D-2D pose estimation.

This creates the migration spine. Later chapters can then plug into the same conventions instead of becoming another collection of standalone scripts.
