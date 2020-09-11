import numpy as np


class _GCPwork:
    def __init__(self):
        pass

    def _point_at_fraction(self, frac):
        """
        Returns (colat, lon) in radians of a point a fraction frac along the minor arc of the GCP
        """
        dist = self.sig01 + frac * self._epicentral_distance()
        colat = self._colat_at_fraction(dist)
        lon = self._lon_at_fraction(dist)
        return (colat, lon)

    def _colat_at_fraction(self, dist):
        numerator = np.cos(self.alpha0) * np.sin(dist)
        denominator = np.sqrt(
            np.cos(dist) ** 2 + np.sin(self.alpha0) ** 2 * np.sin(dist) ** 2
        )
        return np.pi / 2 - np.arctan2(numerator, denominator)

    def _lon_at_fraction(self, dist):
        phi0 = self._node_lon()
        numerator = np.sin(self.alpha0) * np.sin(dist)
        denominator = np.cos(dist)
        return np.arctan2(numerator, denominator) + phi0

    def _epicentral_distance(self):
        """
        Calculates the epicentral distance between the start and stop points
        """
        numerator = np.sqrt(
            (
                np.sin(self.theta1) * np.cos(self.theta2)
                - np.cos(self.theta1) * np.sin(self.theta2) * np.cos(self.phi12)
            )
            ** 2
            + (np.sin(self.theta2) * np.sin(self.phi12)) ** 2
        )
        denominator = np.cos(self.theta1) * np.cos(self.theta2) + np.sin(
            self.theta1
        ) * np.sin(self.theta2) * np.cos(self.phi12)
        return np.arctan2(numerator, denominator)

    def _course_at_start(self):
        """
        Calculates course from start point to stop point
        """
        numerator = np.sin(self.theta2) * np.sin(self.phi12)
        denominator = np.sin(self.theta1) * np.cos(self.theta2) - np.cos(
            self.theta1
        ) * np.sin(self.theta2) * np.cos(self.phi12)
        return np.arctan2(numerator, denominator)

    def _course_at_end(self):
        """
        Calculates course at endpoint
        """
        numerator = np.sin(self.theta1) * np.sin(self.phi12)
        denominator = np.cos(self.theta2) * np.sin(self.theta1) * np.cos(
            self.phi12
        ) - np.cos(self.theta1) * np.sin(self.theta2)
        return np.arctan2(numerator, denominator)

    def _course_at_node(self):
        """
        Calculates course at node (i.e. point at which GC crosses equator northwards)
        """
        numerator = np.sin(self.alpha1) * np.sin(self.theta1)
        denominator = np.sqrt(
            np.cos(self.alpha1) ** 2
            + (np.sin(self.alpha1) ** 2) * (np.cos(self.theta1) ** 2)
        )
        return np.arctan2(numerator, denominator)

    def _node_to_start(self):
        numerator = np.tan(np.pi / 2 - self.theta1)
        denominator = np.cos(self.alpha1)
        return np.arctan2(numerator, denominator)

    def _node_lon(self):
        phi01 = np.arctan2(np.sin(self.alpha0) * np.sin(self.sig01), np.cos(self.sig01))
        return self.start[1] - phi01

    def _endpoints2rad(self):
        """
        Converts endpoints to radians
        Latitudes become colatitudes
        """
        start_lt, start_ln = self.start
        stop_lt, stop_ln = self.stop
        start_clt = 90 - start_lt
        stop_clt = 90 - stop_lt
        self.start = tuple(np.deg2rad((start_clt, start_ln)))
        self.stop = tuple(np.deg2rad((stop_clt, stop_ln)))
