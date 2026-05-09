import numpy as np

from slam.geometry.transforms import make_transform, rotation_matrix_from_rotvec
from slam.io.trajectory import PoseStamped, read_tum_trajectory, write_kitti_trajectory, write_tum_trajectory


def test_tum_trajectory_round_trip(tmp_path):
    transform = make_transform(
        rotation_matrix_from_rotvec(np.array([0.1, -0.2, 0.05])),
        np.array([1.0, 2.0, 3.0]),
    )
    poses = [PoseStamped(timestamp=1.25, transform_wc=transform)]
    path = tmp_path / "trajectory.txt"

    write_tum_trajectory(path, poses)
    loaded = read_tum_trajectory(path)

    assert loaded[0].timestamp == 1.25
    np.testing.assert_allclose(loaded[0].transform_wc, transform, atol=1e-9)


def test_kitti_trajectory_writes_flattened_3x4(tmp_path):
    transform = make_transform(np.eye(3), np.array([1.0, 2.0, 3.0]))
    path = tmp_path / "poses.txt"

    write_kitti_trajectory(path, [transform])

    columns = path.read_text(encoding="utf-8").strip().split()
    assert len(columns) == 12
    np.testing.assert_allclose([float(value) for value in columns], transform[:3, :4].reshape(-1))


def test_read_tum_trajectory_rejects_bad_column_count(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("1 2 3\n", encoding="utf-8")

    try:
        read_tum_trajectory(path)
    except ValueError as exc:
        assert "expected 8 columns" in str(exc)
    else:
        raise AssertionError("expected ValueError")
