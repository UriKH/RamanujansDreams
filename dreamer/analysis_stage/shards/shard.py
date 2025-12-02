"""
Representation of a shard
"""
from dreamer.analysis_stage.shards.hyperplanes import Hyperplane
from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.caching import cached_property
import pulp
import numpy as np
import time
from numba import njit
from scipy.special import gamma, zeta
from dreamer.configs.analysis import analysis_config
from dreamer.utils.types import *


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
            target_yield = int(n_samples * fraction * 1.05) # Try to over sample

            # Edge case: If cone is tiny, ensure we at least look for 1
            if target_yield < 1:
                target_yield = 1
        sampler = CHRRSampler(self.A, np.zeros_like(self.b), R=np.ceil(R*1.5), thinning=5)
        samples, t = sampler.sample(target_yield)
        return {
            Position({sym: v for v, sym in zip(p, self.symbols)})
            for p in samples
        }

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
        res = self.__find_integer_point_milp(
            self.A, self.b_shifted,
            xmin=[-analysis_config.VALIDATION_BOUND_BOX_DIM] * self.dim,
            xmax=[analysis_config.VALIDATION_BOUND_BOX_DIM] * self.dim
        )
        if res is None:
            return None
        return res + self.shift

    @cached_property
    def is_valid(self):
        if self.start_coord is None:
            return False
        d, _ = self.A.shape
        return self.__find_integer_point_milp(
            self.A, np.zeros(d),
            xmin=[-analysis_config.VALIDATION_BOUND_BOX_DIM] * self.dim,
            xmax=[analysis_config.VALIDATION_BOUND_BOX_DIM] * self.dim
        ) is not None


@njit(cache=True)
def gcd_recursive(a, b):
    while b:
        a, b = b, a % b
    return a

@njit(cache=True)
def get_gcd_of_array(arr):
    """Calculates GCD of a vector. Returns 1 immediately if any pair gives 1."""
    d = len(arr)
    if d == 0: return 0
    result = abs(arr[0])
    for i in range(1, d):
        result = gcd_recursive(result, abs(arr[i]))
        if result == 1:
            return 1
    return result

@njit(cache=True)
def is_valid_integer_point(point, A, b, R_sq):
    """
    Checks 3 conditions:
    1. Inside Radius (L2)
    2. Inside Cone (Ax <= b)
    3. GCD == 1
    """
    # 1. Radius Check
    norm_sq = 0.0
    for x in point:
        norm_sq += x*x
    if norm_sq > R_sq or norm_sq == 0: # Exclude origin
        return False

    # 2. Cone Check (Ax <= b)
    # Manual dot product for speed
    m, d = A.shape
    for i in range(m):
        dot = 0.0
        for j in range(d):
            dot += A[i, j] * point[j]
        if dot < b[i]:
            return False

    # 3. GCD Check (Primitive)
    if get_gcd_of_array(point) != 1:
        return False

    return True


@njit(cache=True)
def get_chrr_limits(idx, x, A_cols, b, current_Ax, R_sq):
    """
    Computes the valid range [t_min, t_max] for moving along axis 'idx'
    such that x + t*e_idx stays inside Ax<=b AND ||x|| <= R
    """
    t_min = -1e20 # arbitrary large
    t_max = 1e20

    # --- 1. Cone Constraints (Ax <= b) ---
    # Condition: A_row * (x + t*e_i) <= b_row
    #            (A_row*x) + t*A_row[i] <= b_row
    #            t * A_col_i[row] <= b_row - current_Ax[row]

    # We iterate over the pre-transposed A (A_cols) for cache efficiency
    col_data = A_cols[idx]
    m = len(b)

    for row in range(m):
        slope = col_data[row]
        # rem = b[row] - current_Ax[row] (but slightly safer to recompute if accumulating error?)
        # For CHRR speed, we usually use the tracked Ax.
        rem = b[row] - current_Ax[row]

        if slope > 1e-9:
            # t <= rem / slope
            t_max = min(t_max, rem / slope)
        elif slope < -1e-9:
            # t * neg <= rem  ->  t >= rem / neg
            t_min = max(t_min, rem / slope)
        else:
            # slope is 0. If constraint violated, line is invalid entirely.
            if rem < 0:
                return 1.0, -1.0 # Return empty interval

    # --- 2. Ball Constraint (||x + t*e_i||^2 <= R^2) ---
    # sum(x_j^2) - x_i^2 + (x_i + t)^2 <= R^2
    # (x_i + t)^2 <= R^2 - (current_norm_sq - x_i^2)

    current_norm_sq = 0.0
    for val in x:
        current_norm_sq += val*val

    rem_rad_sq = R_sq - (current_norm_sq - x[idx]**2)

    if rem_rad_sq < 0:
        return 1.0, -1.0 # Invalid

    limit_r = np.sqrt(rem_rad_sq)
    # -limit_r <= x_i + t <= limit_r
    # t >= -limit_r - x_i
    # t <= limit_r - x_i

    t_min = max(t_min, -limit_r - x[idx])
    t_max = min(t_max, limit_r - x[idx])

    return t_min, t_max

@njit(cache=True)
def chrr_walker(A, A_cols, b, R_sq, start_point, n_desired, thinning, buf_out):
    """
    A: (m, d)
    A_cols: (d, m) - Transposed A for fast column access
    """
    m, d = A.shape

    # Current State
    x = start_point.copy()
    current_Ax = np.dot(A, x)

    found = 0
    steps = 0

    # Temp buffer for integer check
    temp_int = np.zeros(d, dtype=np.int64)

    while found < n_desired:
        # 1. Coordinate Hit-and-Run Step
        # Pick random axis
        axis_idx = np.random.randint(0, d)

        # Get valid line segment
        t_min, t_max = get_chrr_limits(axis_idx, x, A_cols, b, current_Ax, R_sq)

        if t_max >= t_min:
            # Move to random point in interval
            t = np.random.uniform(t_min, t_max)

            # Update State
            x[axis_idx] += t
            # Update Ax efficiently: Ax_new = Ax_old + t * A_col_i
            # This is O(m)
            for row in range(m):
                current_Ax[row] += t * A_cols[axis_idx, row]

        steps += 1

        # 2. Harvesting (Thinning)
        # We only try to harvest every 'thinning' steps to ensure mixing
        if steps % thinning == 0:
            # Try to extract an integer point
            # Technique: Take continuous x, add Uniform(-0.5, 0.5), Round.
            # Then CHECK constraints.

            valid_harvest = True
            norm_sq = 0.0

            for k in range(d):
                # Add jitter + Round
                val = x[k] + np.random.uniform(-0.5, 0.5)
                ival = int(round(val))
                temp_int[k] = ival
                norm_sq += ival*ival

            # Check Radius
            if norm_sq > R_sq or norm_sq == 0:
                valid_harvest = False

            # Check Cone (Must re-check because rounding might push us out)
            if valid_harvest:
                for row in range(m):
                    dot = 0.0
                    for col in range(d):
                        dot += A[row, col] * temp_int[col]
                    if dot > b[row]:
                        valid_harvest = False
                        break

            # Check GCD
            if valid_harvest:
                if get_gcd_of_array(temp_int) != 1:
                    valid_harvest = False

            if valid_harvest:
                # Store
                buf_out[found, :] = temp_int[:]
                found += 1

    return steps


class CHRRSampler:
    def __init__(self, A, b, R, thinning=3):
        self.A = np.ascontiguousarray(A, dtype=np.float64)
        # We pre-transpose A for faster column access in the walker
        self.A_cols = np.ascontiguousarray(A.T, dtype=np.float64)
        self.b = np.ascontiguousarray(b, dtype=np.float64)
        self.R = float(R)
        self.R_sq = self.R**2
        self.thinning = thinning

    def find_start_point(self):
        """Finds a valid continuous point inside to start the chain."""
        # Simple Rejection Sampling Attempt (usually finds one instantly)
        # If the cone is SO thin this fails, we need LP, but let's assume this works.
        d = self.A.shape[1]
        for _ in range(10000):
            # Sample in small box around origin or uniform in R
            cand = np.random.uniform(-self.R/10, self.R/10, size=d)
            if np.linalg.norm(cand) > self.R: continue
            if np.all(self.A @ cand <= self.b):
                return cand
        raise RuntimeError("Could not find starting point for CHRR. Cone is too thin or closed.")

    def sample(self, n_samples):
        t0 = time.time()
        start_pt = self.find_start_point()

        # Buffer for Numba
        buf = np.zeros((n_samples, self.A.shape[1]), dtype=np.int64)

        # Run Walker
        # Note: CHRR naturally produces duplicates if the chain stays in the same
        # "integer cell" for multiple harvest steps.
        chrr_walker(self.A, self.A_cols, self.b, self.R_sq, start_pt, n_samples, self.thinning, buf)

        # Post-process for global Uniqueness
        # Because CHRR is a chain, we might need to run longer to get *unique* points
        # So we filter here.
        # unique_set = set(tuple(x) for x in buf)
        unique_set = list(tuple(x) for x in buf)

        # Convert back to array
        final_arr = np.array(list(unique_set))
        return final_arr, time.time() - t0


if __name__ == '__main__':
    a = np.array([[1, 2], [3, 4]])
    b = np.array([1, 1])
    x, y = sp.symbols('x y')
    shard = Shard(a, b, Position({x: 0.5, y: 0.5}), [x, y])
    print(shard.b_shifted)
