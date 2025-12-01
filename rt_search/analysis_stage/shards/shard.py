"""
Representation of a shard
"""
from rt_search.analysis_stage.shards.hyperplanes import Hyperplane
from rt_search.analysis_stage.shards.searchable import *
from rt_search.utils.caching import *
import pulp
from typing import Union
import numpy as np
import time
from numba import njit, float64, int64, boolean
from scipy.special import gamma, zeta
from ...utils.logger import Logger
from scipy.optimize import linprog


class Shard(Searchable):
    def __init__(self,
                 cmf: CMF,
                 constant: str,
                 A: np.ndarray,
                 b: np.array,
                 shift: Position,
                 symbols: List[sp.Symbol]
                 ):
        """
        :param A: Matrix A defining the linear terms in the inequalities
        :param b: Vector b defining the free terms in the inequalities
        :param group: The shard group this shard is part of
        :param shift: The shift in start points required
        :param symbols: Symbols used by the CMF which this shard is part of
        """
        super().__init__(cmf, constant)
        self.A = A
        self.b = b
        self.symbols = symbols
        self.shift = np.array([shift[sym] for sym in self.symbols])

    def in_space(self, point: Position) -> bool:
        point = np.array(point.sorted().values())
        return np.all(self.A @ point < self.b)

    def get_interior_point(self) -> Position:
        return Position({sym: v for v, sym in zip(self.start_coord, self.symbols)})

    @Logger.log_exec
    def sample_trajectories(self, n_samples, *, strict: Optional[bool] = False) -> Set[Position]:
        """
        Sample trajectories from the shard.
        :param n_samples: number of samples to generate
        :param strict: if compute as n_samples, else compute n_samples * fraction.
        (fraction of the cone is taking from the sphere)
        :return: a set of sampled trajectories
        """
        # R, fraction = self.compute_required_radius(self.A, n_samples, sample_count=500)
        # if strict:
        #     # Compute the radius with some safety mesures
        #     n_samples = np.ceil(int(n_samples / fraction * 1.02))
        #     R = self.compute_ball_radius(len(self.symbols), n_samples)
        # else:
        #     n_samples = int(np.ceil(n_samples * fraction))

        def _estimate_cone_fraction(A, n_trials=5000):
            """Helper to estimate what % of the sphere is covered by the cone."""
            d = A.shape[1]
            # Sample random directions
            raw = np.random.normal(size=(n_trials, d))
            # Normalize
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            dirs = raw / norms
            # Check Ax <= 0 (Cone angular check)
            projections = dirs @ A.T
            inside = np.all(projections <= 1e-9, axis=1)

            frac = np.mean(inside)

            # Safety for extremely thin cones to avoid division by zero
            if frac == 0:
                return 1.0 / n_trials  # Conservative lower bound

            return frac
        # 1. Estimate Cone Fraction (Monte Carlo)
        # We strip this out of the radius function to use it for logic decisions
        fraction = _estimate_cone_fraction(self.A)

        if strict:
            # CASE: We need EXACTLY n_samples inside the cone.

            # We need to find an R such that Volume(Cone_R) contains n_samples.
            # N_cone = N_sphere * fraction
            # Therefore: N_sphere_equivalent = n_samples / fraction

            # SAFETY MARGIN: This is critical.
            # If our fraction estimate is off by 1%, we might miss the target.
            # We pad the target by 20% (1.2) to ensure the volume is definitely large enough.
            n_target_safe = int((n_samples / fraction) * 1.2)

            # Compute R for this expanded sphere count
            R = self.compute_ball_radius(len(self.symbols), n_target_safe)

            # We tell the sampler to stop exactly when it hits n_samples
            target_yield = n_samples

        else:
            # CASE: We treat n_samples as the "Density" of the full sphere.
            # We just want whatever naturally falls into the cone.

            # R is calculated for the sphere density
            R = self.compute_ball_radius(len(self.symbols), n_samples)

            # The expected number of points is roughly N * fraction.
            # We don't force the sampler to find more than this.
            target_yield = int(n_samples * fraction * 0.95)

            # Edge case: If cone is tiny, ensure we at least look for 1
            if target_yield < 1:
                target_yield = 1
        sampler = CHRRSampler(self.A, np.zeros_like(self.b), R=R+2, thinning=5)
        print(f'sampling: {target_yield} points in cone fraction: {fraction} and radius {R}\nmatrix: {self.A}\n')
        return {
            Position({sym: v for v, sym in zip(p, self.symbols)})
            for p in sampler.sample(target_yield)[0]
        }

    @staticmethod
    def compute_required_radius(A, n_target, sample_count=10_000, safety_factor=1.05):
        """
        Estimates the Radius R needed to get n_target integer points
        with GCD=1 inside the cone Ax <= 0.
        """
        m, d = A.shape

        # 1. Estimate Fraction of Sphere Covered by Cone (Monte Carlo)
        # We ignore 'b' here and use b=0 because for large R,
        # the volume is dominated by the angular opening (Ax <= 0).

        # Generate random directions on unit sphere
        raw = np.random.normal(size=(sample_count, d))
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        dirs = raw / norms

        # Check how many satisfy Ax <= 0
        # A: (m, d), dirs: (N, d) -> (N, m)
        projections = dirs @ A.T

        # Check if all constraints <= 0 for each direction
        inside_mask = np.all(projections <= 1e-9, axis=1)
        fraction = np.sum(inside_mask) / sample_count

        if fraction == 0:
            raise ValueError("Cone is too thin! Monte Carlo found 0 hits. "
                             "Cannot estimate R. You might need to pick R manually or increase sample_count.")

        print(f"Estimated Cone Fraction: {fraction:.4%}")

        # 2. Volume of Unit Ball in d-dimensions
        # V_unit = pi^(d/2) / gamma(d/2 + 1)
        vol_unit_ball = (np.pi ** (d / 2.0)) / gamma(d / 2.0 + 1.0)

        # 3. Density of Primitive Vectors (GCD=1)
        # Density approx 1/zeta(d)
        # For d=1, density is technically 0 in limit, but practical for geometry.
        prim_density = 1.0 / zeta(d) if d > 1 else 1.0

        # 4. Solve for R
        # N = Vol_Unit * R^d * Fraction * Density
        # R^d = N / (Vol_Unit * Fraction * Density)

        term = n_target / (vol_unit_ball * fraction * prim_density)
        R_est = term ** (1.0 / d)

        # Apply safety factor
        R_final = R_est * safety_factor

        return R_final, fraction * 100

    @staticmethod
    def compute_ball_radius(d, n_samples):
        """
        Computes the Radius R required for a d-dimensional ball to contain
        approximately n_samples integer points.

        Parameters:
        -----------
        d : int
            Number of dimensions.
        n_samples : int
            Target number of points.
        primitive : bool
            If True, adjusts for GCD=1 requirement (points must be primitive).
            If False, counts all integer points.

        Returns:
        --------
        float : The estimated Radius R.
        """
        # 1. Volume of a Unit Ball in d-dimensions
        # V_unit = pi^(d/2) / Gamma(d/2 + 1)
        vol_unit_ball = (np.pi ** (d / 2.0)) / gamma(d / 2.0 + 1.0)

        # 2. Density Adjustment
        # If we want GCD=1, the density of points is 1/zeta(d).
        # Note: zeta(1) is infinity, but physically 1D density is usually treated as 1 for counts
        density = 1.0 / zeta(d) if d > 1 else 1.0

        # 3. Solve for R
        # N = Vol_Unit * R^d * Density
        # R = ( N / (Vol_Unit * Density) ) ^ (1/d)

        term = n_samples / (vol_unit_ball * density)
        R = term ** (1.0 / d)

        return R

    @staticmethod
    # @lru_cache
    def __find_integer_point_milp(
            A: np.array, b: np.array, xmin: Optional[List[int]] = None, xmax: Optional[List[int]] = None
    ) -> Optional[np.array]:
        """
        Use PuLP MILP CBC solver to find feasible point
        :param A: Original hyperplane constraints (linear terms)
        :param b: Original hyperplane constraints (free terms)
        :param xmin: minimum bound on each variable
        :param xmax: maximum bound on each variable
        :return: Vector representing the feasible point
        """
        m, d = A.shape
        prob = pulp.LpProblem('find_int_point', pulp.LpStatusOptimal)
        vars = [
            pulp.LpVariable(
                f'x{i}',
                lowBound=int(xmin[i]) if xmin is not None else None,
                upBound=int(xmax[i]) if xmax is not None else None,
                cat='Integer'
            )
            for i in range(d)
        ]

        # no objective, just feasibility: add 0 objective
        prob += 0
        for i in range(m):
            prob += pulp.lpSum(A[i, j] * vars[j] for j in range(d)) <= b[i] - 1e-6
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[prob.status] != 'Optimal':
            return None
        return np.array([int(val) if (val := var.value()) else 0 for var in vars], dtype=int)

    @staticmethod
    def generate_matrices(
            hyperplanes: List[Hyperplane],
            above_below_indicator: Union[List[int], Tuple[int, ...]]
    ) -> Tuple[np.ndarray, np.array, List[sp.Symbol]]:
        # if (l_hps := len(hyperplanes)) != (l_ind := 2**len(above_below_indicator)):
        #     raise ValueError(f"Number of hyperplanes does not match number of indicators {l_hps}!={l_ind}")
        if any(ind != 1 and ind != -1 for ind in above_below_indicator):
            raise ValueError(f"Indicators vector must be 1 (above) or -1 (below)")

        symbols = hyperplanes[0].symbols
        symbols = list(symbols)
        vectors = []
        free_terms = []

        for expr, ind in zip(hyperplanes, above_below_indicator):
            if isinstance(expr, Hyperplane):
                hp = expr
            else:
                hp = Hyperplane(expr, symbols)
            if ind == 1:
                v, free = hp.as_above_vector
            else:
                v, free = hp.as_below_vector
            free_terms.append(free)
            vectors.append(v)
        return np.vstack(tuple(vectors)), np.array(free_terms), symbols

    @property
    def b_shifted(self):
        """
        Computes b with respect to shifted hyperplanes: Ax <= b' instead of Ax <= b
        :return: The shifted b vector
        """
        S = np.eye(self.shift.shape[0]) * self.shift
        return self.b + (self.A @ S).sum(axis=1)

    @cached_property
    def start_coord(self):
        res = self.__find_integer_point_milp(self.A, self.b_shifted, xmin=[-5] * 3, xmax=[5] * 3)
        if res is None:
            return None
        return res + self.shift

    @cached_property
    def is_valid(self):
        if self.start_coord is None:
            return False
        d, _ = self.A.shape
        return self.__find_integer_point_milp(self.A, np.zeros(d), xmin=[-5] * 3, xmax=[5] * 3) is not None

# --- Numba Logic ---

@njit(cache=True)
def gcd_recursive(a, b):
    while b:
        a, b = b, a % b
    return a


@njit(cache=True)
def get_gcd_of_array(arr):
    d = len(arr)
    if d == 0: return 0
    result = abs(arr[0])
    for i in range(1, d):
        result = gcd_recursive(result, abs(arr[i]))
        if result == 1: return 1
    return result


@njit(cache=True)
def get_chrr_limits(idx, x, A_cols, b, current_Ax, R_sq):
    t_min = -1e20
    t_max = 1e20
    col_data = A_cols[idx]
    m = len(b)

    for row in range(m):
        slope = col_data[row]
        # Tolerance: Allow being slightly "off" the wall but not deep outside
        # We allow 1e-12 float error.
        rem = b[row] - current_Ax[row] + 1e-12

        if slope > 1e-9:
            t_max = min(t_max, rem / slope)
        elif slope < -1e-9:
            t_min = max(t_min, rem / slope)
        else:
            if rem < -1e-9: return 1.0, -1.0

    current_norm_sq = 0.0
    for val in x: current_norm_sq += val * val
    rem_rad_sq = R_sq - (current_norm_sq - x[idx] ** 2)

    if rem_rad_sq < 0:
        return 1.0, -1.0

    limit_r = np.sqrt(rem_rad_sq)
    t_min = max(t_min, -limit_r - x[idx])
    t_max = min(t_max, limit_r - x[idx])

    return t_min, t_max


@njit(cache=True)
def chrr_walker(A, A_cols, b, R_sq, start_point, n_desired, thinning, buf_out):
    m, d = A.shape
    x = start_point.copy()

    # Initialize Ax accurately
    current_Ax = np.zeros(m)
    for r in range(m):
        dot = 0.0
        for c in range(d):
            dot += A[r, c] * x[c]
        current_Ax[r] = dot

    found = 0
    steps = 0
    max_steps = n_desired * thinning * 5000

    temp_int = np.zeros(d, dtype=np.int64)

    while found < n_desired and steps < max_steps:
        # 1. Walk
        axis_idx = np.random.randint(0, d)
        t_min, t_max = get_chrr_limits(axis_idx, x, A_cols, b, current_Ax, R_sq)

        if t_max >= t_min:
            t = np.random.uniform(t_min, t_max)
            x[axis_idx] += t
            for row in range(m):
                current_Ax[row] += t * A_cols[axis_idx, row]

        steps += 1

        # --- FIX 1: Prevent Drift (CRITICAL) ---
        # Every 100 steps, recompute Ax from scratch to clear floating point errors.
        if steps % 100 == 0:
            for r in range(m):
                dot = 0.0
                for c in range(d):
                    dot += A[r, c] * x[c]
                current_Ax[r] = dot

        # 2. Harvest
        if steps % thinning == 0:
            valid_harvest = True
            norm_sq = 0.0

            # Spatial Dithering (Uniform -0.5 to 0.5)
            for k in range(d):
                val = x[k] + np.random.uniform(-0.5, 0.5)
                ival = int(round(val))
                temp_int[k] = ival
                norm_sq += ival * ival

            if norm_sq > R_sq or norm_sq == 0:
                valid_harvest = False

            if valid_harvest:
                for row in range(m):
                    dot = 0.0
                    for col in range(d):
                        dot += A[row, col] * temp_int[col]

                    # --- CHECK: Strictness ---
                    # You requested Au < 0.
                    # We check: If dot >= b - 1e-9, then REJECT.
                    # Since b=0, this rejects anything >= -1e-9.
                    # This ensures we only keep strictly negative dots (Au < 0).
                    if dot >= b[row] - 1e-9:
                        valid_harvest = False
                        break

            if valid_harvest and get_gcd_of_array(temp_int) != 1:
                valid_harvest = False

            if valid_harvest:
                buf_out[found, :] = temp_int[:]
                found += 1

    return steps


class CHRRSampler:
    def __init__(self, A, b, R, thinning=3):
        A_float = np.array(A, dtype=np.float64)
        b_float = np.array(b, dtype=np.float64)
        norms = np.linalg.norm(A_float, axis=1)
        # Handle potential zero-rows safely
        norms[norms == 0] = 1.0

        # Store normalized versions
        self.A = (A_float / norms[:, None])
        self.b = (b_float / norms)
        # self.A = np.ascontiguousarray(A, dtype=np.float64)
        # We pre-transpose A for faster column access in the walker
        self.A_cols = np.ascontiguousarray(A.T, dtype=np.float64)
        # self.b = np.ascontiguousarray(b, dtype=np.float64)
        self.R = float(R)
        self.R_sq = self.R**2
        self.thinning = thinning

    def find_start_point(self):
        """Finds a valid continuous point inside using LP (Robust for thin cones)."""
        d = self.A.shape[1]

        # Use Highs solver for stability
        # We look for a point x s.t. Ax <= b
        res = linprog(c=np.zeros(d), A_ub=self.A, b_ub=self.b,
                      bounds=(-self.R, self.R), method='highs')

        if res.success:
            pt = res.x
            norm = np.linalg.norm(pt)
            if norm < self.R:
                return pt
            else:
                return pt * (0.99 * self.R / norm)

        # Fallback to random if LP fails (rare)
        for _ in range(5000):
            cand = np.random.uniform(-self.R / 10, self.R / 10, size=d)
            if np.linalg.norm(cand) > self.R: continue
            if np.all(self.A @ cand <= self.b - 1e-9):
                return cand

        raise RuntimeError("Could not find starting point for CHRR. Cone is too thin or closed.")

    def sample(self, n_samples):
        t0 = time.time()
        start_pt = self.find_start_point()
        buf = np.zeros((n_samples, self.A.shape[1]), dtype=np.int64)

        chrr_walker(self.A, self.A_cols, self.b, self.R_sq, start_pt, n_samples, self.thinning, buf)

        # Filter 0s (in case walker didn't finish) and duplicates
        unique_set = list(set(tuple(x) for x in buf if not np.all(x == 0)))

        return np.array(unique_set), time.time() - t0


if __name__ == '__main__':
    a = np.array([[1, 2], [3, 4]])
    b = np.array([1, 1])
    x, y = sp.symbols('x y')
    shard = Shard(a, b, Position({x: 0.5, y: 0.5}), [x, y])
    print(shard.b_shifted)
