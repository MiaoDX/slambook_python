# Chapter Port Status

Status terms:

- `Baseline complete`: the chapter has importable Python modules, at least one runnable CLI example, and core tests for the migrated behavior.
- `Legacy retained`: original root-level scripts remain available while the migrated package and examples are the preferred path.

| Chapter | Topic | Status | Baseline backend | Optional backend | Notes |
| --- | --- | --- | --- | --- | --- |
| ch2 | Python/project basics | Baseline complete | Python packaging | none | Package scaffold, optional extras, pytest config, and core-only CI added |
| ch3 | Rigid body motion | Baseline complete | NumPy/SciPy | pytransform3d | Homogeneous transform helper and CLI example added |
| ch4 | Lie groups | Baseline complete | local SO3/SE3 | jaxlie/spatialmath | Minimal exp/log helper and CLI example added |
| ch5 | Camera and images | Baseline complete | OpenCV | Open3D | Pinhole/stereo helpers, distortion coefficients, image basics CLI, and RGB-D point cloud CLI added |
| ch6 | Nonlinear optimization | Baseline complete | SciPy | GTSAM | SciPy curve fitting example with robust loss coverage added |
| ch7 | Feature VO | Baseline complete | OpenCV | LightGlue/LoFTR/PyCOLMAP | ORB/SIFT interfaces, 2D-2D/3D-2D/3D-3D pose examples, triangulation, LightGlue/LoFTR adapters, PyCOLMAP reference reconstruction CLI, optional Rerun match/track logging, and legacy script status documented |
| ch8 | Direct VO | Baseline complete | OpenCV/SciPy | JAX | LK flow, sparse 2D/SE3 residual refinement CLIs, image pyramid, interpolation helpers, and photometric tests added |
| ch9 | VO project | Baseline complete | OpenCV/SciPy | GTSAM/PyCOLMAP | Data model, frame feature extraction, VO coordinator, depth-assisted local-map runner, local-map tracking with motion-only BA, optional PyCOLMAP absolute-pose reference, trajectory/viz helpers, documented image layout, and smoke-tested monocular runner added |
| ch10 | Bundle adjustment | Baseline complete | SciPy | GTSAM/PyCOLMAP | BAL parser, included tiny BAL fixture, `scipy_ba` module, SciPy/GTSAM CLI backends, PyCOLMAP BAL pose-reference CLI, and optional backend guards added |
| ch11 | Pose graph | Baseline complete | SciPy small graph | GTSAM | g2o parser, included tiny fixture, SciPy/GTSAM optimizer paths, CLI trajectory export, Matplotlib plot export, optional Rerun logging, and GTSAM guard added |
| ch12 | Loop closure | Baseline complete | OpenCV descriptors/BoW | FAISS/hloc-style comparison | OpenCV descriptor builder, BoW vocabulary training, NumPy/FAISS retrieval, temporal filtering, and CLIs added |
| ch13 | Dense mapping | Baseline complete | Open3D-compatible arrays/PLY/occupancy | Open3D/Rerun | RGB-D arrays, known-pose fusion, voxel occupancy export, voxel downsampling, normal estimation, known-pose monocular depth estimation, optional Open3D mesh reconstruction, PLY output/downsample/Rerun CLI, and optional viz guards added |

The migration baseline is complete for the planned chapter surface. Remaining
work is follow-on hardening: validating examples on full real datasets and
expanding optional backend tests on machines where GTSAM, PyCOLMAP, LightGlue,
FAISS, Open3D, and Rerun are installed.
