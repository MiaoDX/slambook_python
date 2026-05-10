# Chapter Port Status

| Chapter | Topic | Status | Baseline backend | Optional backend | Notes |
| --- | --- | --- | --- | --- | --- |
| ch2 | Python/project basics | In progress | Python packaging | none | Package scaffold and core-only CI added |
| ch3 | Rigid body motion | In progress | NumPy/SciPy | pytransform3d | Homogeneous transform helper and CLI example added |
| ch4 | Lie groups | In progress | local SO3/SE3 | jaxlie/spatialmath | Minimal exp/log helper and CLI example added |
| ch5 | Camera and images | In progress | OpenCV | Open3D | Pinhole/stereo helpers and image basics CLI added |
| ch6 | Nonlinear optimization | In progress | SciPy | GTSAM | SciPy curve fitting example added |
| ch7 | Feature VO | In progress | OpenCV | LightGlue/PyCOLMAP | ORB/SIFT interfaces, learned matcher guards, two-view helpers, and ch7 CLIs added |
| ch8 | Direct VO | In progress | OpenCV/SciPy | JAX | LK flow and direct sparse residual CLIs plus image pyramid/interpolation helpers added |
| ch9 | VO project | In progress | OpenCV | GTSAM/PyCOLMAP | Data model, config loading, trajectory/viz helpers, and minimal monocular runner added |
| ch10 | Bundle adjustment | In progress | SciPy | GTSAM/PyCOLMAP | BAL parser, `scipy_ba` module, CLI, and optional backend guards added |
| ch11 | Pose graph | In progress | SciPy small graph | GTSAM | g2o parser, SciPy optimizer, CLI, and optional GTSAM guard added |
| ch12 | Loop closure | In progress | OpenCV descriptors | FAISS/hloc | OpenCV descriptor builder, retrieval, temporal filtering, and CLIs added |
| ch13 | Dense mapping | In progress | Open3D | Rerun | RGB-D arrays, PLY output, single-frame CLI, and optional viz guards added |
