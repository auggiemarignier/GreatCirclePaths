import numpy as np
from scipy import sparse
import pyssht

from greatcirclepaths.gcpsampling import _MWGCP


def Y22(thetas, phis):
    theta, phi = np.meshgrid(thetas, phis, indexing="ij")
    norm = (1 / 4) * np.sqrt(15 / (2 * np.pi))
    return norm * np.sin(theta) ** 2 * np.exp(2j * phi)


def Y22_int_theta(theta, phi):
    """Path integral for constant theta"""
    norm = (-1j / 8) * np.sqrt(15 / (2 * np.pi))
    return norm * np.sin(theta) ** 3 * (np.exp(2j * phi) - 1)


def Y1010(thetas, phis):
    theta, phi = np.meshgrid(thetas, phis, indexing="ij")
    norm = (1 / 1024) * np.sqrt(969969 / np.pi)
    return norm * np.sin(theta) ** 10 * np.exp(10j * phi)


def Y1010_int_theta(theta, phi):
    """Path integral for constant theta"""
    norm = (-1j / 10240) * np.sqrt(969969 / np.pi)
    return norm * np.sin(theta) ** 11 * (np.exp(10j * phi) - 1)


def build_path_matrix_ser(start, stop, L):
    paths = np.zeros((start.shape[0], pyssht.sample_length(L, Method="MW")))
    for i, (stt, stp) in enumerate(zip(start, stop)):
        path = _MWGCP(
            stt,
            stp,
            L=L,
            weighting="distances",
            latlon=True,
        )
        path.get_points(points_per_rad=150)
        path.fill()
        paths[i] = path.map
    return sparse.csr_matrix(paths)
