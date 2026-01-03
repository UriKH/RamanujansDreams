import time
from numba import njit
import numpy as np
from servicemanager import LogErrorMsg
from sympy.logic.algorithms.dpll2 import Level

from dreamer.utils.logger import Logger
from dreamer.utils.types import *


@njit(cache=True)
def gcd_recursive(a: int, b: int) -> int:
    """
    Computes GCD of a and b
    """
    while b:
        a, b = b, a % b
    return a


@njit(cache=True)
def get_gcd_of_array(arr) -> int:
    """
    Calculates GCD of a vector.
    Returns 1 immediately if any pair gives 1.
    """
    d = len(arr)
    if d == 0:
        return 0
    result = abs(arr[0])
    for i in range(1, d):
        result = gcd_recursive(result, abs(arr[i]))
        if result == 1:
            return 1
    return result


@njit(cache=True)
def is_valid_integer_point(point, A, b, R_sq) -> bool:
    """
    Checks if a point is inside the cone and radius, and has coordinates gcd = 1.
    :param point: A point to validate
    :param A: Matrix representing the cone
    :param b: Vector representing the cone's upper bound
    :param R_sq: maximum norm of the point
    :return:
    """
    # Radius check
    norm_sq = 0.0
    for x in point:
        norm_sq += x * x
    if norm_sq > R_sq or norm_sq == 0:  # Exclude origin
        return False

    # Cone check (Ax < b)
    # Manual dot product for speed
    m, d = A.shape
    for i in range(m):
        dot = 0.0
        for j in range(d):
            dot += A[i, j] * point[j]
        if dot < b[i]:
            return False

    # GCD check
    return get_gcd_of_array(point) == 1


@njit(cache=True)
def get_chrr_limits(idx, x, A_cols, b, current_Ax, R_sq) -> Tuple[float, float]:
    """
    Computes the valid range [t_min, t_max] for moving along axis 'idx'
    such that:
     * x + t * e_idx stays inside Ax <= b
     * ||x|| <= R
    :param idx: axis index
    :param x: a vector x
    :param A_cols: A transposed matrix
    :param b: the b vector representing the cone upper bound
    :param current_Ax: A @ x multiplication result
    :param R_sq: Maximum norm of the point x
    :return: the range [t_min, t_max]
    """
    t_min = -1e20
    t_max = 1e20

    # Cone constraints (Ax <= b)
    # Condition: A_row * (x + t*e_i) <= b_row
    #            (A_row*x) + t*A_row[i] <= b_row
    #            t * A_col_i[row] <= b_row - current_Ax[row]
    col_data = A_cols[idx]
    m = len(b)

    for row in range(m):
        slope = col_data[row]
        rem = b[row] - current_Ax[row]

        if slope > 1e-9:
            t_max = min(t_max, rem / slope)
        elif slope < -1e-9:
            t_min = max(t_min, rem / slope)
        else:
            # slope is 0. If constraint violated, line is invalid entirely.
            if rem < 0:
                return 1., -1.    # empty interval

    # Ball constraint (||x + t * e_i||^2 <= R^2)
    # sum(x_j^2) - x_i^2 + (x_i + t)^2 <= R^2
    # (x_i + t)^2 <= R^2 - (current_norm_sq - x_i^2)
    current_norm_sq = np.dot(x, x)
    rem_rad_sq = R_sq - (current_norm_sq - x[idx] ** 2)
    if rem_rad_sq < 0:
        return 1., -1.  # empty interval

    # -limit_r <= x_i + t <= limit_r    --->    t >= -limit_r - x_i and t <= limit_r - x_i
    limit_r = np.sqrt(rem_rad_sq)
    t_min = max(t_min, -limit_r - x[idx])
    t_max = min(t_max, limit_r - x[idx])
    return t_min, t_max


@njit(cache=True)
def chrr_walker(A, A_cols, b, R_sq, start_point, n_desired, thinning, buf_out, max_steps):
    """
    A: (m, d)
    A_cols: (d, m) - Transposed A for fast column access
    """
    ERROR_BOUND = 1e-8
    m, d = A.shape
    x = start_point.copy()
    current_Ax = np.dot(A, x)
    found = 0
    steps = 0

    while found < n_desired and steps < max_steps:
        # coordinate Hit-and-Run step
        axis_idx = np.random.randint(0, d)  # Pick random axis
        t_min, t_max = get_chrr_limits(axis_idx, x, A_cols, b, current_Ax, R_sq)
        if t_max >= t_min:
            t = np.random.uniform(t_min, t_max)
            x[axis_idx] += t
            # Update: Ax_new = Ax_old + t * A_col_i
            for row in range(m):
                current_Ax[row] += t * A_cols[axis_idx, row]
        steps += 1

        # Harvesting (+ thinning)
        if steps % thinning == 0:
            # Try to extract an integer point
            # Technique: Take continuous x, add Uniform(-0.5, 0.5), Round.
            temp = x + np.random.uniform(-0.5, 0.5, d)
            sample = np.round(temp)
            sample.astype(np.int64)
            norm_sq = np.dot(sample, sample)
            if norm_sq > R_sq or norm_sq == 0:
                continue

            # Check Cone (Must re-check because rounding might push us out)
            if np.any(np.dot(A, sample) > b - ERROR_BOUND):
                continue

            if get_gcd_of_array(sample) != 1:
                continue

            # Store
            buf_out[found, :] = sample[:]
            found += 1
    return found, steps


class CHRRSampler:
    def __init__(self, A, b, R, thinning=3, start=None):
        """
        Continuous Hierarchical Random Walker Sampler inside a cone Ax < b intersecting ball with radius R.
        :param A: Matrix representing the cone
        :param b: Vector representing the cone's upper bound
        :param R: Radius of the ball
        :param thinning: Measure of mixing - one of how many valid points to choose
        :param start: Starting point for the sampler.
        """
        self.A = np.ascontiguousarray(A, dtype=np.float64)
        self.A_cols = np.ascontiguousarray(A.T, dtype=np.float64)
        self.b = np.ascontiguousarray(b, dtype=np.float64)
        self.R = float(R)
        self.R_sq = self.R ** 2
        self.thinning = thinning
        self.start = start

    def find_start_point(self):
        """
        Numerically finds a valid continuous point inside to start the chain.
        """
        # Simple Rejection Sampling Attempt (usually finds one instantly)
        # If the cone is SO thin this fails, we need LP, but let's assume this works.
        if self.start is not None:
            return self.start
        d = self.A.shape[1]
        for _ in range(10000):
            # Sample in small box around origin or uniform in R
            cand = np.random.uniform(-self.R / 10, self.R / 10, d)
            if np.linalg.norm(cand) > self.R:
                continue
            if np.all(self.A @ cand < self.b):
                return cand
        raise RuntimeError("Could not find starting point for CHRR. Cone is too thin or closed.")

    def sample(self, n_samples):
        """
        Sample points inside a cone.
        :param n_samples: Number of points in space, sample only the part which are inside the cone
        :return: A set of points inside the cone with GCD of coordinates = 1
        """
        t0 = time.time()
        try:
            start_pt = self.find_start_point()
        except RuntimeError as e:
            Logger(f'RuntimeError in sampler: {e}', Logger.Levels.warning).log()
            return [], 0

        max_steps_per_round = max(2000 * n_samples * self.thinning, 100_000)

        buf = np.zeros((n_samples, self.A.shape[1]), dtype=np.int64)

        retries = 0
        total_found = 0
        while total_found < n_samples and retries < 5:
            needed = n_samples
            found_now, _ = chrr_walker(
                self.A, self.A_cols, self.b, self.R_sq,
                start_pt, needed, self.thinning,
                buf[total_found:], max_steps_per_round
            )

            total_found += found_now
            if total_found < n_samples:
                # The Radius is likely too small: expand Radius by 50%
                self.R *= 1.5
                self.R_sq = self.R ** 2
                retries += 1

        if retries == 5:
            Logger(
                f'Number of trajectories is too small, try increasing radius or number of retries'
                f' (THIS COULD BE A BUG - CONTACT DEV)', Logger.Levels.warning
            ).log(msg_prefix='\n')
        unique_set = set(tuple(x) for x in buf)
        final_arr = np.array(list(unique_set))
        return final_arr, time.time() - t0
