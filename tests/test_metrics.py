import numpy as np

from bead_rl.metrics import points_to_polyline_distance, valid_segments


def test_point_to_horizontal_segment():
    line = np.array([[0.0, 0.0], [2.0, 0.0]])
    points = np.array([[1.0, 1.0], [3.0, 0.0], [0.5, 0.0]])
    distances = points_to_polyline_distance(points, line)
    np.testing.assert_allclose(distances, [1.0, 1.0, 0.0], atol=1e-12)


def test_path_break_removes_bridge_segment():
    line = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]])
    starts, ends = valid_segments(line, path_breaks={2})
    assert len(starts) == 2
    assert not np.any(np.all(starts == [1.0, 0.0], axis=1) & np.all(ends == [10.0, 0.0], axis=1))
    distance = points_to_polyline_distance(np.array([[5.0, 0.0]]), line, path_breaks={2})
    np.testing.assert_allclose(distance, [4.0], atol=1e-12)


def test_empty_polyline_is_infinite():
    out = points_to_polyline_distance(np.array([[0.0, 0.0]]), np.empty((0, 2)))
    assert np.isinf(out[0])
