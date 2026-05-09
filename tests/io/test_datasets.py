from pathlib import Path

import pytest

from slam.io.datasets import associate_tum_rgbd, list_image_sequence


def test_list_image_sequence_sorts_matching_files(tmp_path):
    (tmp_path / "b.png").write_text("", encoding="utf-8")
    (tmp_path / "a.png").write_text("", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("", encoding="utf-8")

    frames = list_image_sequence(tmp_path)

    assert [frame.index for frame in frames] == [0, 1]
    assert [frame.image_path.name for frame in frames] == ["a.png", "b.png"]


def test_list_image_sequence_rejects_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_image_sequence(tmp_path / "missing")


def test_associate_tum_rgbd_matches_nearest_depth_timestamp(tmp_path):
    (tmp_path / "rgb").mkdir()
    (tmp_path / "depth").mkdir()
    rgb_txt = tmp_path / "rgb.txt"
    depth_txt = tmp_path / "depth.txt"
    rgb_txt.write_text(
        "\n".join(
            [
                "# timestamp filename",
                "1.000 rgb/1.png",
                "2.000 rgb/2.png",
                "3.000 rgb/3.png",
            ]
        ),
        encoding="utf-8",
    )
    depth_txt.write_text(
        "\n".join(
            [
                "0.991 depth/1.png",
                "2.015 depth/2.png",
                "3.100 depth/3.png",
            ]
        ),
        encoding="utf-8",
    )

    frames = associate_tum_rgbd(rgb_txt, depth_txt, max_difference=0.02)

    assert len(frames) == 2
    assert [frame.timestamp for frame in frames] == [1.0, 2.0]
    assert [frame.depth_timestamp for frame in frames] == [0.991, 2.015]
    assert [frame.rgb_path.name for frame in frames] == ["1.png", "2.png"]
    assert [frame.depth_path.name for frame in frames] == ["1.png", "2.png"]
