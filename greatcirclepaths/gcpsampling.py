import numpy as np
import healpy as hp
import pyssht

from greatcirclepaths.gcpwork import _GCPwork


class _HpxGCP(_GCPwork):
    def __init__(self, start, stop, Nside):
        super().__init__(start, stop)
        self.Nside = Nside
        self.map = np.zeros(hp.nside2npix(Nside))

    def fill(self):
        pixels = [hp.ang2pix(self.Nside, *point) for point in self.points]
        self.map[pixels] = 1


class _MWGCP(_GCPwork):
    def __init__(self, start, stop, L, weighting):
        super().__init__(start, stop)
        self.L = L
        self.map = np.zeros(pyssht.sample_shape(L))
        if weighting:
            self.areas = self.calc_pixel_areas(L)

    def fill(self):
        pixels = [
            (
                pyssht.theta_to_index(point[0], self.L),
                pyssht.phi_to_index(point[1], self.L),
            )
            for point in self.points
        ]
        for pix in pixels:
            self.map[pix] = 1
        if hasattr(self, "areas"):
            self.map *= self.areas
        self.map = self.map.flatten()

    def calc_pixel_areas(self, L, r=1):
        thetas, phis = pyssht.sample_positions(L)
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

    def _pixel_area(self, r, theta1, theta2, phi1, phi2):
        return r ** 2 * (np.cos(theta1) - np.cos(theta2)) * (phi2 - phi1)

    def _polar_cap_area(self, r, alpha):
        return 2 * np.pi * r ** 2 * (1 - np.cos(alpha))
