import pyssht
import pytest
import numpy as np

from greatcirclepaths.gcpsampling import _MWGCP
from .utils import Y1010, Y1010_int_theta, build_path_matrix_ser


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
    path = _MWGCP(start, stop, L, latlon=True)
    areas = path.calc_pixel_areas()
    assert np.isclose(np.sum(areas), 4 * np.pi)


@pytest.mark.parametrize("nearest", [False, True])
def test_mw_path_pixel_distances(random_start_stop, L, nearest):
    start, stop = random_start_stop
    path = _MWGCP(start, stop, L, latlon=True)
    path.get_points(points_per_rad=150)
    pixels = path.select_pixels(nearest)
    distances = path.calc_segment_distances(pixels)
    assert np.isclose(np.sum(distances), path._epicentral_distance())


@pytest.mark.parametrize("nearest", [False, True])
def test_path_integral(L, nearest):
    thetas, phis = pyssht.sample_positions(L, Method="MW")
    f_mw = Y1010(thetas, phis).flatten()
    phi2 = np.linspace(0, np.pi, 300, endpoint=False)

    theta = np.pi / 2
    starts = np.array([[theta, 0] for _ in phi2])
    stops = np.array([[theta, lon] for lon in phi2])
    path_matrix = build_path_matrix_ser(starts, stops, L, nearest)

    numerical = path_matrix.dot(f_mw)
    analytical = Y1010_int_theta(theta, np.radians(phi2)).flatten()

    assert np.allclose(numerical, analytical)


@pytest.mark.parametrize(
    "value, expected", [(6.13, 38), (6.28, 0), (0.05, 0), (6.4, 1), (-np.pi / 4, 34)]
)  # correct for L=20, may fail otherwise
def test_nearest_sample(L, value, expected):
    _, phis = pyssht.sample_positions(L)
    nearest = _MWGCP._nearest_sample(phis, value, wrap_value=2 * np.pi)

    assert nearest == expected
