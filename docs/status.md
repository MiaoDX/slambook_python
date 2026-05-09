# Chapter Port Status

| Chapter | Topic | Status | Baseline backend | Optional backend | Notes |
| --- | --- | --- | --- | --- | --- |
| ch2 | Python/project basics | In progress | Python packaging | none | Package scaffold added |
| ch3 | Rigid body motion | Not started | NumPy/SciPy | pytransform3d | Needed by all later chapters |
| ch4 | Lie groups | Not started | local SO3/SE3 | jaxlie/spatialmath | Keep minimal |
| ch5 | Camera and images | Not started | OpenCV | Open3D | Current stereo script overlaps |
| ch6 | Nonlinear optimization | Not started | SciPy | GTSAM | Curve fitting first |
| ch7 | Feature VO | In progress | OpenCV | LightGlue/PyCOLMAP | ORB/SIFT interfaces, two-view helpers, and 2D-2D CLI added |
| ch8 | Direct VO | Not started | OpenCV/SciPy | JAX | Do after ch6 |
| ch9 | VO project | Not started | OpenCV | GTSAM/PyCOLMAP | Main package milestone |
| ch10 | Bundle adjustment | Not started | SciPy | GTSAM/PyCOLMAP | Needs BAL parser |
| ch11 | Pose graph | Not started | SciPy small graph | GTSAM | Needs g2o parser |
| ch12 | Loop closure | Not started | OpenCV descriptors | FAISS/hloc | Descriptor source explicit |
| ch13 | Dense mapping | Not started | Open3D | Rerun | Replace PCL/Octomap concepts |
