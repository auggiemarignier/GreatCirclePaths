from greatcirclepaths.gcpsampling import _HpxGCP, _MWGCP


class GreatCirclePath:
    """
    Finds all the points along a great circle path and the corresponding pixels
    start/stop are tuples (lat, lon) in degrees
    """
    def __new__(cls, start, stop, sampling, L=None, Nside=None, weighting=None):
        if sampling == "MW":
            return _MWGCP(start, stop, L, weighting)
        elif sampling == "hpx":
            return _HpxGCP(start, stop, Nside)
        else:
            raise ValueError("Invalid sampling method.  Please choose either 'MW' or 'hpx'.")
