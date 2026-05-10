import importlib

import numpy as np
import pytest


def test_faiss_index_flat_l2_round_trip():
    faiss = pytest.importorskip("faiss", reason="install the modern extra for faiss-cpu")
    index = faiss.IndexFlatL2(2)
    vectors = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)

    index.add(vectors)
    distances, indices = index.search(np.array([[0.5, 0.0]], dtype=np.float32), 2)

    assert indices.tolist() == [[0, 1]]
    np.testing.assert_allclose(distances, [[0.25, 2.25]], atol=1e-6)


def test_pycolmap_exposes_reconstruction_pipeline_api():
    pycolmap = pytest.importorskip("pycolmap", reason="install the modern extra for pycolmap")

    assert hasattr(pycolmap, "extract_features")
    assert hasattr(pycolmap, "incremental_mapping")


def test_open3d_point_cloud_api_accepts_numpy_vectors():
    open3d = pytest.importorskip("open3d", reason="install the 3d extra for Open3D")
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(np.array([[0.0, 1.0, 2.0]], dtype=np.float64))

    assert len(point_cloud.points) == 1


def test_rerun_points3d_archetype_can_be_constructed():
    rerun = pytest.importorskip("rerun", reason="install the modern extra for rerun-sdk")
    points = rerun.Points3D(np.array([[0.0, 0.0, 0.0]], dtype=np.float32))

    assert points is not None


def test_gtsam_pose3_api_is_available():
    gtsam = pytest.importorskip("gtsam", reason="install the backend extra for GTSAM")
    pose = gtsam.Pose3()

    assert np.asarray(pose.matrix()).shape == (4, 4)


def test_jax_and_jaxlie_basic_group_api():
    jax = pytest.importorskip("jax", reason="install the backend extra for JAX")
    jaxlie = pytest.importorskip("jaxlie", reason="install the backend extra for jaxlie")

    vector = jax.numpy.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(np.asarray(vector), [1.0, 2.0, 3.0])
    assert jaxlie.SO3.identity().wxyz.shape == (4,)


def test_torch_kornia_lightglue_imports_are_compatible():
    torch = pytest.importorskip("torch", reason="install the modern extra for torch")
    kornia = pytest.importorskip("kornia", reason="install the modern extra for kornia")
    lightglue = pytest.importorskip("lightglue", reason="install the modern extra for LightGlue")

    tensor = torch.as_tensor([[1.0, 2.0]])
    assert tuple(tensor.shape) == (1, 2)
    assert hasattr(kornia, "feature")
    assert hasattr(lightglue, "LightGlue")


def test_evo_imports_for_trajectory_evaluation():
    evo = pytest.importorskip("evo", reason="install the 3d extra for evo")
    trajectory_module = importlib.import_module("evo.core.trajectory")

    assert evo is not None
    assert hasattr(trajectory_module, "PoseTrajectory3D")
