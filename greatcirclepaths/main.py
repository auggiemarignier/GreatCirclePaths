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
    parser.add_argument(
        "method", choices=["MW", "hpx"], help="Sampling method.  Either 'MW' or 'hpx'."
    )
    parser.add_argument(
        "-P", "--processes", type=int, default=4, help="Number of parallel processes"
    )
    parser.add_argument(
        "-N",
        "--npoints",
        type=int,
        help="Number of points to be sampled along each path",
    )
    parser.add_argument(
        "-PPR",
        "--points_per_rad",
        type=int,
        default=200,
        help="Number of points per radian along each path",
    )
    parser.add_argument(
        "-Ns", "--nside", type=int, default=32, help="Healpix Nside parameter",
    )
    parser.add_argument(
        "-L", "--L", type=int, default=32, help="Angular bandlimit",
    )
    parser.add_argument(
        "-W",
        "--weighting",
        choices=["areas", "distances", "average"],
        help="Weight pixels either by area or (average) distance travelled by path",
    )
    parser.add_argument("--latlon", action="store_true", help="Use if inputs are in lat/lon format")
    args = parser.parse_args()

    def build_path(start, stop):
        path = GreatCirclePath(
            start,
            stop,
            args.method,
            Nside=args.nside,
            L=args.L,
            weighting=args.weighting,
            latlon=args.latlon,
        )
        path.get_points(npoints=args.npoints, points_per_rad=args.points_per_rad)
        path.fill()
        return path.map

    def build_path_matrix_par(start, stop, processes):
        itrbl = [(start, stop) for (start, stop) in zip(start, stop)]
        with Pool(processes) as p:
            result = p.starmap_async(build_path, itrbl)
            paths = result.get()
        return sparse.csr_matrix(paths)

    def build_path_matrix_ser(start, stop):
        buf = build_path(start[0], stop[0])
        paths = np.zeros((start.shape[0], buf.shape[0]))
        for i, (stt, stp) in enumerate(zip(start, stop)):
            paths[i] = build_path(stt, stp)
        return sparse.csr_matrix(paths)

    def build_path_matrix(start, stop, processes=4):
        if processes > 0:
            path_matrix = build_path_matrix_par(start, stop, processes)
        else:
            path_matrix = build_path_matrix_ser(start, stop)
        return path_matrix

    all_data = np.loadtxt(args.datafile)
    start_lat = all_data[:, 0]
    start_lon = all_data[:, 1]
    start = np.stack([start_lat, start_lon], axis=1)
    stop_lat = all_data[:, 2]
    stop_lon = all_data[:, 3]
    stop = np.stack([stop_lat, stop_lon], axis=1)

    path_matrix = build_path_matrix(start, stop, processes=args.processes)
    sparse.save_npz(args.outfile, path_matrix)
