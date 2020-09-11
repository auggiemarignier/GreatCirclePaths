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

    def get_points(self, npoints):
        # TODO: figure out how to choose npoints based on pathlength and Nside
        fracs = np.linspace(0, 1, npoints)
        self.points = [self._point_at_fraction(frac) for frac in fracs]


class _MWGCP(_GCPwork):
    def __init__(self, start, stop, L):
        super().__init__(start, stop)
        self.L = L
        self.map = np.zeros(pyssht.sample_shape(L))

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
        self.map = self.map.flatten()

    def get_points(self, npoints):
        # TODO: figure out how to choose npoints based on pathlength and Nside
        fracs = np.linspace(0, 1, npoints)
        self.points = [self._point_at_fraction(frac) for frac in fracs]
