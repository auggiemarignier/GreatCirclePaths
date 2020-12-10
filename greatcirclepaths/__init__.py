from greatcirclepaths.gcpsampling import _HpxGCP, _MWGCP
from greatcirclepaths.gcpwork import _GCPwork


class GreatCirclePath:
    """
    Finds all the points along a great circle path and the corresponding pixels
    start/stop are tuples (colat, lon) in radians.
    If latlon=True start/stop are tuples (lat, lon) in degrees
    """

    def __new__(
        cls,
        start,
        stop,
        sampling=None,
        L=None,
        Nside=None,
        weighting=None,
        latlon=False,
    ):
        if sampling == "MW":
            return _MWGCP(start, stop, L, weighting, latlon=latlon)
        elif sampling == "hpx":
            return _HpxGCP(start, stop, Nside, latlon=latlon)
        elif sampling is None:
            return _GCPwork(start, stop, latlon=latlon)
        else:
            raise ValueError(
                "Invalid sampling method.  Please choose either 'MW' or 'hpx'."
            )
