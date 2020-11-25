import pyssht
import pytest
import numpy as np

from greatcirclepaths.gcpsampling import _MWGCP
from .utils import Y22, Y22_int_theta, build_path_matrix_ser


@pytest.fixture
def L():
    return 20


@pytest.fixture
def random_start_stop():
    from random import uniform

    start = (uniform(-90, 90), uniform(-180, 180))
    stop = (uniform(-90, 90), uniform(-180, 180))
    return start, stop


def test_mw_pixel_areas(random_start_stop, L):
    start, stop = random_start_stop
    path = _MWGCP(start, stop, L)
    areas = path.calc_pixel_areas()
    assert np.isclose(np.sum(areas), 4 * np.pi)


def test_mw_path_pixel_distances(random_start_stop, L):
    start, stop = random_start_stop
    path = _MWGCP(start, stop, L)
    path.get_points(points_per_rad=150)
    pixels = [
        (pyssht.theta_to_index(point[0], L), pyssht.phi_to_index(point[1], L),)
        for point in path.points
    ]
    distances = path.calc_segment_distances(pixels)
    assert np.isclose(np.sum(distances), path._epicentral_distance())


def test_path_integral(L):
    thetas, phis = pyssht.sample_positions(L, Method="MW")
    f_mw = Y22(thetas, phis).flatten()
    path_lengths = np.linspace(0, 180, 300, endpoint=False)

    theta = thetas[pyssht.theta_to_index(np.pi / 2, L)]
    starts = np.array([[np.degrees(np.pi / 2 - theta), 0] for _ in path_lengths])
    stops = np.array([[np.degrees(np.pi / 2 - theta), lon] for lon in path_lengths])
    path_matrix = build_path_matrix_ser(starts, stops, L)

    numerical = path_matrix.dot(f_mw)
    analytical = Y22_int_theta(theta, np.radians(path_lengths)).flatten()

    assert np.allclose(numerical, analytical)
