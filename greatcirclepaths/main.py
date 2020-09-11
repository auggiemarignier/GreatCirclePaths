from greatcirclepaths import GreatCirclePath
from multiprocessing import Pool
from scipy import sparse
import numpy as np


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "datafile",
        type=str,
        help="File with columns start_lat, start_lon, end_lat, end_lon",
    )
    parser.add_argument("outfile", type=str, help="output filename for sparse matrix")
    parser.add_argument("method", choices=["MW", "hpx"], help="Sampling method.  Either 'MW' or 'hpx'.")
    parser.add_argument(
        "-P", "--processes", type=int, default=4, help="Number of parallel processes"
    )
    parser.add_argument(
        "-N",
        "--npoints",
        type=int,
        default=1000,
        help="Number of points to be sampled along each path",
    )
    parser.add_argument(
        "-Ns", "--nside", type=int, default=32, help="Healpix Nside parameter",
    )
    parser.add_argument(
        "-L", "--L", type=int, default=32, help="Angular bandlimit",
    )
    args = parser.parse_args()

    def build_path(start, stop):
        path = GreatCirclePath(start, stop, args.method, Nside=args.nside, L=args.L)
        path.get_points(args.npoints)
        path.fill()
        return path.map

    def build_path_matrix(start, stop, processes=4):
        itrbl = [(start, stop) for (start, stop) in zip(start, stop)]
        with Pool(processes) as p:
            result = p.starmap_async(build_path, itrbl)
            paths = result.get()
        return sparse.csr_matrix(paths)

    all_data = np.loadtxt(args.datafile)
    start_lat = all_data[:, 0]
    start_lon = all_data[:, 1]
    start = np.stack([start_lat, start_lon], axis=1)
    stop_lat = all_data[:, 2]
    stop_lon = all_data[:, 3]
    stop = np.stack([stop_lat, stop_lon], axis=1)

    path_matrix = build_path_matrix(start, stop, processes=args.processes)
    sparse.save_npz(args.outpath, path_matrix)
