import numpy as np
import pyssht
import pytest

from greatcirclepaths.gcpwork import _GCPwork
from greatcirclepaths.gcpsampling import _MWGCP


# Tests based on example given on Great Circle Navigation Wikipedia page


@pytest.fixture
def path():
    start = (-33, -71.6)
    stop = (31.4, 121.8)
    return _GCPwork(start, stop)


def test_gcp_course(path):
    assert np.round(np.rad2deg(path._course_at_start()), 2) == -94.41
    assert np.round(np.rad2deg(path._course_at_end()), 2) == -78.42
    assert np.round(np.rad2deg(path._course_at_node()), 2) == -56.74


def test_gcp_epicentral_distance(path):
    assert np.round(np.rad2deg(path._epicentral_distance()), 2) == 168.56
    assert np.round(np.rad2deg(path._node_to_start()), 2) == -96.76


def test_gcp_midpoint(path):
    colat, lon = np.rad2deg(path._point_at_fraction(0.5))
    lat = 90 - colat
    assert np.round(lat, 2) == -6.81
    assert np.round(lon, 2) == -159.18


@pytest.fixture
def random_start_stop():
    from random import uniform
    start = (uniform(-90, 90), uniform(-180, 180))
    stop = (uniform(-90, 90), uniform(-180, 180))
    return start, stop


def test_mw_pixel_areas(random_start_stop):
    start, stop = random_start_stop
    path = _MWGCP(start, stop, 20)
    areas = path.calc_pixel_areas()
    assert np.isclose(np.sum(areas), 4 * np.pi)


def test_mw_path_pixel_distances(random_start_stop):
    start, stop = random_start_stop
    path = _MWGCP(start, stop, 20)
    path.get_points(100)
    pixels = [
        (
            pyssht.theta_to_index(point[0], path.L),
            pyssht.phi_to_index(point[1], path.L),
        )
        for point in path.points
    ]
    distances = path.calc_segment_distances(pixels)
    assert np.isclose(np.sum(distances), path._epicentral_distance())
