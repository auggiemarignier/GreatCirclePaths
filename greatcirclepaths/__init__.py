from greatcirclepaths.gcpsampling import _HpxGCP


class GreatCirclePath:
    def __new__(self, start, stop, sampling, L=None, Nside=None):
        if sampling == "MW":
            pass
        elif sampling == "hpx":
            return _HpxGCP(start, stop, Nside)
        else:
            raise ValueError("Invalid sampling method.  Please choose either 'MW' or 'hpx'.")
