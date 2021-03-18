import numpy as np
import healpy as hp
import pyssht

from greatcirclepaths.gcpwork import _GCPwork


class _HpxGCP(_GCPwork):
    def __init__(self, start, stop, Nside, latlon=False):
        super().__init__(start, stop, latlon=latlon)
        self.Nside = Nside
        self.map = np.zeros(hp.nside2npix(Nside))

    def fill(self):
        pixels = [hp.ang2pix(self.Nside, *point) for point in self.points]
        self.map[pixels] = 1


class _MWGCP(_GCPwork):
    def __init__(self, start, stop, L, weighting=None, latlon=False):
        if weighting not in [None, "areas", "distances"]:
            raise ValueError("weighting must be either None, 'areas' or 'distances")
        super().__init__(start, stop, latlon=latlon)
        self.L = L
        self.map = np.zeros(pyssht.sample_shape(L))
        self.weighting = weighting

    def fill(self):
        samples = self.select_samples()
        for samp in samples:
            self.map[samp] = 1
        if self.weighting is not None:
            if self.weighting == "areas":
                weights = self.calc_pixel_areas()
            if self.weighting == "distances":
                weights = self.calc_segment_distances(samples)
            self.map *= weights
        self.map = self.map.flatten()

    def select_samples(self, n_nearest=1):
        """
        n_nearest is the number of nearest samples wanted in theta AND phi.
        e.g if n_nearest = 2, 4 points will be returned
        """
        thetas, phis = pyssht.sample_positions(self.L)
        samples = [
            (
                self._nearest_samples(thetas, point[0], n=n_nearest),
                self._nearest_samples(
                    phis, point[1], n=n_nearest, wrap_value=2 * np.pi
                ),
            )
            for point in self.points
        ]
        return samples

    def calc_segment_distances(self, samples):
        """
        samples is a list of tuples (theta_ind, phi_ind) of MW samples that the path goes past
        """
        distances = np.zeros(pyssht.sample_shape(self.L, Method="MW"))
        for point1, point2, samp1, samp2 in zip(
            self.points, self.points[1:], samples, samples[1:]
        ):
            seg = _GCPwork(point1, point2)
            if samp1 == samp2:
                distances[samp1] += seg._epicentral_distance()
            else:
                pointhalf = seg._point_at_fraction(0.5)
                seg1 = _GCPwork(point1, pointhalf)
                seg2 = _GCPwork(pointhalf, point2)
                distances[samp1] += seg1._epicentral_distance()
                distances[samp2] += seg2._epicentral_distance()
        return distances

    def calc_pixel_areas(self, r=1):
        thetas, phis = pyssht.sample_positions(self.L)
        nthetas, nphis = thetas.shape[0], phis.shape[0]
        areas = np.zeros((nthetas, nphis), dtype=np.float64)
        phis = np.append(phis, [2 * np.pi])
        areas[0] = self._polar_cap_area(r, thetas[0]) / nphis
        for t, theta1 in enumerate(thetas[:-1]):
            theta2 = thetas[t + 1]
            for p, phi1 in enumerate(phis[:-1]):
                phi2 = phis[p + 1]
                areas[t + 1][p] = self._pixel_area(r, theta1, theta2, phi1, phi2)
        return areas

    @staticmethod
    def _pixel_area(r, theta1, theta2, phi1, phi2):
        return r ** 2 * (np.cos(theta1) - np.cos(theta2)) * (phi2 - phi1)

    @staticmethod
    def _polar_cap_area(r, alpha):
        return 2 * np.pi * r ** 2 * (1 - np.cos(alpha))

    @staticmethod
    def _nearest_samples(arr, v, n=1, wrap_value=None):
        if wrap_value is None:
            diff = np.abs(arr - v)
            return np.argpartition(diff, n)[:n]
        else:
            if v < 0:
                v += wrap_value
            if v > wrap_value:
                v -= wrap_value
            arr = np.concatenate([arr, wrap_value], axis=None)
            diff = np.abs(arr - v)
            inds = np.argpartition(diff, n)[:n]
            inds[inds == len(arr) - 1] = 0
            return inds
