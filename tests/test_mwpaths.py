import pyssht
import pytest
import numpy as np

from greatcirclepaths.gcpsampling import _MWGCP


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


@pytest.mark.parametrize("n", [1, 2])
def test_mw_path_pixel_distances(random_start_stop, L, n):
    start, stop = random_start_stop
    path = _MWGCP(start, stop, L, latlon=True)
    path.get_points(points_per_rad=150)
    pixels = path.select_samples(n_nearest=n)
    distances = path.calc_segment_distances(pixels)
    start = np.radians(start)
    stop = np.radians(stop)
    lawofcosines = np.arccos(
        np.sin(start[0]) * np.sin(stop[0])
        + np.cos(start[0]) * np.cos(stop[0]) * np.cos(np.abs(start[1] - stop[1]))
    )
    assert np.isclose(np.sum(distances), path._epicentral_distance())
    assert np.isclose(path._epicentral_distance(), lawofcosines)


@pytest.mark.parametrize(
    "value, expected", [(6.13, 38), (6.28, 0), (0.05, 0), (6.4, 1), (-np.pi / 4, 34)]
)  # correct for L=20, may fail otherwise
def test_nearest_sample(L, value, expected):
    _, phis = pyssht.sample_positions(L)
    nearest = _MWGCP._nearest_samples(phis, value, wrap_value=2 * np.pi)
    assert nearest == expected


@pytest.mark.parametrize("value, expected", [(np.pi / 2, [9, 10])])
def test_nearest_n_samples(L, value, expected):
    thetas, _ = pyssht.sample_positions(L)
    nearest = _MWGCP._nearest_samples(thetas, value, n=2)
    assert np.alltrue(nearest == expected)


@pytest.mark.parametrize("value, expected", [(6.283, [0, 38])])
def test_nearest_n_samples_wrap(L, value, expected):
    _, phis = pyssht.sample_positions(L)
    nearest = _MWGCP._nearest_samples(phis, value, n=2, wrap_value=2 * np.pi)
    assert np.alltrue(nearest == expected)


@pytest.mark.parametrize("n", [1, 2])
def test_distance_point_to_sample_weights(L, n):
    from random import uniform

    point = (uniform(0, np.pi), uniform(0, 2 * np.pi))
    thetas, phis = pyssht.sample_positions(L)
    nearest_thetas = _MWGCP._nearest_samples(thetas, point[0], n=n)
    nearest_phis = _MWGCP._nearest_samples(phis, point[1], n=n, wrap_value=2 * np.pi)
    weights = _MWGCP._point_to_sample_weights(point, nearest_thetas, nearest_phis, L)
    assert np.isclose(np.sum(weights), 1)
    nz = np.nonzero(weights)
    assert all([tt in nz[0] for tt in nearest_thetas]) and all(
        [nzt in nearest_thetas for nzt in nz[0]]
    )
    assert all([pp in nz[1] for pp in nearest_phis]) and all([
        nzp in nearest_phis for nzp in nz[1]
    ])
