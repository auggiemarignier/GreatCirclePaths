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
    def __init__(self, start, stop, L, weighting=None):
        if weighting not in [None, "areas", "distances"]:
            raise ValueError("wieghting must be either None, 'areas' or 'distances")
        super().__init__(start, stop)
        self.L = L
        self.map = np.zeros(pyssht.sample_shape(L))
        self.weighting = weighting

    def fill(self, nearest=False):
        pixels = self.select_pixels(nearest)
        for pix in pixels:
            self.map[pix] = 1
        if self.weighting is not None:
            if self.weighting == "areas":
                weights = self.calc_pixel_areas()
            if self.weighting == "distances":
                weights = self.calc_segment_distances(pixels)
            self.map *= weights
        self.map = self.map.flatten()

    def select_pixels(self, nearest=False):
        if not nearest:
            pixels = [
                (
                    pyssht.theta_to_index(point[0], self.L),
                    pyssht.phi_to_index(point[1], self.L),
                )
                for point in self.points
            ]
        else:
            thetas, phis = pyssht.sample_positions(self.L)
            pixels = [
                (
                    self._nearest_sample(thetas, point[0]),
                    self._nearest_sample(phis, point[1]),
                )
                for point in self.points
            ]
        return pixels

    def calc_segment_distances(self, pixels):
        """
        pixels is a list of tuples (theta_ind, phi_ind) through which
        the path passes
        """
        distances = np.zeros(pyssht.sample_shape(self.L, Method="MW"))
        for point1, point2, pix1, pix2 in zip(
            self.points, self.points[1:], pixels, pixels[1:]
        ):
            point1 = self._colatlonrad2latlondeg(*point1)
            point2 = self._colatlonrad2latlondeg(*point2)
            seg = self.__class__(point1, point2, self.L, weighting=None)
            if pix1 == pix2:
                distances[pix1] += seg._epicentral_distance()
            else:
                pointhalf = self._colatlonrad2latlondeg(*seg._point_at_fraction(0.5))
                seg1 = self.__class__(point1, pointhalf, self.L, weighting=None)
                seg2 = self.__class__(pointhalf, point2, self.L, weighting=None)
                distances[pix1] += seg1._epicentral_distance()
                distances[pix2] += seg2._epicentral_distance()
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
    def _colatlonrad2latlondeg(colatr, lonr):
        latr = np.pi / 2 - colatr
        latd = np.rad2deg(latr)
        lond = np.rad2deg(lonr)
        return latd, lond

    @staticmethod
    def _nearest_sample(arr, v):
        return np.argmin(np.abs(arr - v))
