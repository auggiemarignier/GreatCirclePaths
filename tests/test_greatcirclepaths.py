import numpy as np
from pytest_cases import parametrize_with_cases, fixture

from greatcirclepaths import GreatCirclePath


# Tests based on example given on Great Circle Navigation Wikipedia page

def case_sampling_hp():
    return ("hpx", 32, None)


def case_sampling_mw():
    return ("MW", None, 32)


@fixture
@parametrize_with_cases("sampling", cases=[case_sampling_hp, case_sampling_mw])
def path(sampling):
    start = (-33, -71.6)
    stop = (31.4, 121.8)
    sample = sampling[0]
    Nside = sampling[1]
    L = sampling[2]
    return GreatCirclePath(start, stop, sample, Nside=Nside, L=L)


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
