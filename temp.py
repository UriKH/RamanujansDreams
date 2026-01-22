import itertools
import sympy as sp
import numpy as np
from time import time
from numba import njit, prange


@njit
def substitute_linear(hps: np.ndarray, point: np.ndarray):
    buf = np.zeros(hps.shape[0])
    for i in range(hps.shape[0]):
        for j in range(point.shape[0]):
            buf[i] += hps[i, j] * point[j]
    return buf


if __name__ == "__main__":

    points = list(itertools.product(tuple(list(range(-2, 3))), repeat=6))

    start = time()
    points_np = np.array(points, dtype=np.float64)
    shift_np = np.array([-1, sp.Rational(1, 2), 0]*2, dtype=np.float64)
    points_np += shift_np
    end = time()
    print(f'{end - start:.6f} time to compute points fast?')

    start = time()
    points = [
        tuple(coord + shift for coord, shift in zip(p, [-1, sp.Rational(1, 2), 0] * 2)) for p in points
    ]
    end = time()
    print(f'{end - start:.6f} time to compute points slow?')

