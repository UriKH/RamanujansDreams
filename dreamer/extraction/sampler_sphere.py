import numpy as np
from scipy.special import gamma, zeta

from .numba_utils import *


@njit(cache=True)
def check_points(points, R_sq):
    """
    Filters a batch of points.
    :return: A boolean mask: true if (Norm <= R and GCD == 1) else false
    """
    n, d = points.shape
    mask = np.zeros(n, dtype=np.bool_)

    for i in range(n):
        norm_sq = 0.0
        for j in range(d):
            norm_sq += points[i, j] * points[i, j]
        if norm_sq > R_sq or norm_sq == 0:
            continue

        if get_gcd_of_array(points[i]) == 1:
            mask[i] = True

    return mask


class PrimitiveSphereSampler:
    """
    Utility class for sampling primitive points in a hypersphere.
    """

    def __init__(self, d, batch_size=100_000):
        """
        :param d: dimensions of the sphere
        :param batch_size: number of points to sample per batch
        """
        self.d = d
        self.batch_size = batch_size
        self.rng = np.random.default_rng()

    def compute_radius(self, n_samples):
        """
        Calculates the minimal Radius R to contain n_samples primitive points.
        :param n_samples: Number of primitive points in the hypersphere.
        :return: The hypersphere radius.
        """
        # Vol_unit = pi^(d/2) / gamma(d/2 + 1)
        vol_unit_ball = (np.pi ** (self.d / 2.0)) / gamma(self.d / 2.0 + 1.0)
        density = 1.0 / zeta(self.d) if self.d > 1 else 1.0
        R = (n_samples / (vol_unit_ball * density)) ** (1.0 / self.d)
        return np.ceil(R * 1.2)   # make sure to take a big enough buffer

    def sample(self, n_samples):
        """
        :param n_samples: Number of points to sample (divided to batches).
        :return: The sampled points
        """
        R = self.compute_radius(n_samples)
        R_sq = R * R

        print(f"Sampling {n_samples} primitive points in {self.d}D Sphere (R={R:.2f})...")

        collected = set()

        while len(collected) < n_samples:
            # generate candidates
            raw = self.rng.standard_normal((self.batch_size, self.d))
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            dirs = raw / norms
            u = self.rng.random((self.batch_size, 1))
            radii = R * (u ** (1.0 / self.d))
            continuous_pts = dirs * radii
            noise = self.rng.uniform(-0.5, 0.5, size=(self.batch_size, self.d))
            candidates = np.round(continuous_pts + noise).astype(np.int64)

            # filtering
            mask = check_points(candidates, R_sq)
            valid_batch = candidates[mask]

            # update
            for p in valid_batch:
                collected.add(tuple(p))
                if len(collected) >= n_samples:
                    break
        return np.array(list(collected))
