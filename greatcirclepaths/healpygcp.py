import numpy as np
import healpy as hp

from greatcirclepaths.gcpwork import _GCPwork


class _HpxGCP(_GCPwork):
    def __init__(self, start, stop, Nside):
        """
        Finds all the points along a great circle path and the corresponding healpix pixels
        start/stop are tuples (lat, lon) in degrees
        """
        self.start = start
        self.stop = stop
        self.Nside = Nside
        self.map = np.zeros(hp.nside2npix(Nside))

        self._endpoints2rad()
        self.theta1 = self.start[0]
        self.theta2 = self.stop[0]
        self.phi12 = self.stop[1] - self.start[1]
        if self.phi12 > 180:
            self.phi12 -= 360
        elif self.phi12 < -180:
            self.phi12 += 360
        else:
            pass
        self.alpha1 = self._course_at_start()
        self.alpha0 = self._course_at_node()
        self.sig01 = self._node_to_start()

    def fill(self):
        pixels = [hp.ang2pix(self.Nside, *point) for point in self.points]
        self.map[pixels] = 1

    def get_points(self, npoints):
        # TODO: figure out how to choose npoints based on pathlength and Nside
        fracs = np.linspace(0, 1, npoints)
        self.points = [self._point_at_fraction(frac) for frac in fracs]
