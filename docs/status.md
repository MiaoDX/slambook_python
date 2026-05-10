# Chapter Port Status

| Chapter | Topic | Status | Baseline backend | Optional backend | Notes |
| --- | --- | --- | --- | --- | --- |
| ch2 | Python/project basics | In progress | Python packaging | none | Package scaffold and core-only CI added |
| ch3 | Rigid body motion | In progress | NumPy/SciPy | pytransform3d | Homogeneous transform helper and CLI example added |
| ch4 | Lie groups | In progress | local SO3/SE3 | jaxlie/spatialmath | Minimal exp/log helper and CLI example added |
| ch5 | Camera and images | In progress | OpenCV | Open3D | Pinhole/stereo helpers, distortion coefficients, and image basics CLI added |
| ch6 | Nonlinear optimization | In progress | SciPy | GTSAM | SciPy curve fitting example added |
| ch7 | Feature VO | In progress | OpenCV | LightGlue/PyCOLMAP | ORB/SIFT interfaces, learned matcher guards, ch7 CLIs, and legacy script status documented |
| ch8 | Direct VO | In progress | OpenCV/SciPy | JAX | LK flow, sparse 2D/SE3 residual refinement CLIs, image pyramid, and interpolation helpers added |
| ch9 | VO project | In progress | OpenCV | GTSAM/PyCOLMAP | Data model, frame feature extraction, VO coordinator, local-map matching/tracking, trajectory/viz helpers, documented image layout, and smoke-tested monocular runner added |
| ch10 | Bundle adjustment | In progress | SciPy | GTSAM/PyCOLMAP | BAL parser, included tiny BAL fixture, `scipy_ba` module, SciPy CLI, PyCOLMAP reference guard, and optional backend guards added |
| ch11 | Pose graph | In progress | SciPy small graph | GTSAM | g2o parser, included tiny fixture, SciPy/GTSAM optimizer paths, CLI trajectory export, Matplotlib plot export, optional Rerun logging, and GTSAM guard added |
| ch12 | Loop closure | In progress | OpenCV descriptors | FAISS/hloc | OpenCV descriptor builder, NumPy/FAISS retrieval, temporal filtering, and CLIs added |
| ch13 | Dense mapping | In progress | Open3D | Rerun | RGB-D arrays, known-pose fusion, voxel downsampling, normal estimation, PLY output/downsample/Rerun CLI, and optional viz guards added |
