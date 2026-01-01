"""
Representation of a shard
"""
from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.caching import cached_property
from dreamer.utils.logger import Logger
import numpy as np
import time
from numba import njit
from scipy.special import gamma, zeta
from dreamer.utils.types import *
from dreamer.utils.constants.constant import Constant

from scipy.optimize import linprog
from scipy.optimize import milp, LinearConstraint, Bounds


class Shard(Searchable):
    def __init__(self,
                 cmf: CMF,
                 constant: Constant,
                 A: np.ndarray,
                 b: np.array,
                 shift: Position,
                 symbols: List[sp.Symbol],
                 start_coord: Optional[Position] = None
                 ):
        """
        :param A: Matrix A defining the linear terms in the inequalities
        :param b: Vector b defining the free terms in the inequalities
        :param group: The shard group this shard is part of
        :param shift: The shift in start points required
        :param symbols: Symbols used by the CMF which this shard is part of
        """
        super().__init__(cmf, constant, shift)
        self.A = A
        self.b = b
        self.symbols = symbols
        self.shift = np.array([shift[sym] for sym in self.symbols])
        self.start_coord = start_coord

    def in_space(self, point: Position) -> bool:
        point = np.array(point.sorted().values())
        return np.all(self.A @ point < self.b)

    def get_interior_point(self) -> Position:
        start = self.start_coord
        return Position({sym: v for v, sym in zip(start.values(), self.symbols)})

    def sample_trajectories(self, n_samples, *, strict: Optional[bool] = False) -> Set[Position]:
        """
        Sample trajectories from the shard.
        :param n_samples: number of samples to generate
        :param strict: if compute as n_samples, else compute n_samples * fraction.
        (fraction of the cone is taking from the sphere)
        :return: a set of sampled trajectories
        """
        def _estimate_cone_fraction(A, n_trials=5000):
            """Helper to estimate what % of the sphere is covered by the cone."""
            d = A.shape[1]
            raw = np.random.normal(size=(n_trials, d))
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            dirs = raw / norms
            projections = dirs @ A.T
            inside = np.all(projections <= 1e-9, axis=1)

            frac = np.mean(inside)

            # Safety for extremely thin cones to avoid division by zero
            if frac == 0:
                return 1.0 / n_trials  # Conservative lower bound
            return frac

        # with Logger.simple_timer(f'compute fraction of cone'):
        fraction = _estimate_cone_fraction(self.A)

        # with Logger.simple_timer(f'compute radius of cone'):
        if strict:
            # CASE: We need EXACTLY n_samples inside the cone.
            n_target_safe = int((n_samples / fraction) * 1.2)
            R = self.compute_ball_radius(len(self.symbols), n_target_safe)
            target_yield = n_samples
        else:
            # CASE: We treat n_samples as the "Density" of the full sphere.
            R = self.compute_ball_radius(len(self.symbols), n_samples)
            # Over sample a small amount in case fraction was underestimated
            target_yield = int(n_samples * fraction * 1.05)

            # Edge case: If cone is tiny, ensure we at least look for 1
            if target_yield < 1:
                target_yield = 1

        # print(f'do sampling ... (requested {target_yield})')
        # with Logger.simple_timer(f'sample trajectories using CHRR [sampled = {target_yield}]'):
        sampler = CHRRSampler(self.A, np.zeros_like(self.b), R=np.ceil(R + 0.5), thinning=5, start=np.array(list(self.get_interior_point().values()), dtype=np.float64))
        samples, t = sampler.sample(target_yield)
        return {
            Position({sym: sp.sympify(v) for v, sym in zip(p, self.symbols)})
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
        vol_unit_ball = (np.pi ** (d / 2.0)) / gamma(d / 2.0 + 1.0)
        density = 1.0 / zeta(d) if d > 1 else 1.0
        term = n_samples / (vol_unit_ball * density)
        R = term ** (1.0 / d)

        return R

    @staticmethod
    def find_integer_feasible_point(A: np.ndarray, b: np.ndarray) -> np.ndarray | None:
        """
        Determines if there is an INTEGER solution x such that Ax <= b.

        Args:
            A (np.ndarray): Matrix coefficients.
            b (np.ndarray): Bounds vector.

        Returns:
            tuple: (is_feasible, point)
        """
        n_vars = A.shape[1]

        # 1. Objective: We just want feasibility, so coefficients are 0.
        c = np.zeros(n_vars)

        # 2. Constraints: Ax <= b
        # milp uses the form: lb <= A.dot(x) <= ub
        # So we set lb = -infinity, ub = b
        constraints = LinearConstraint(A, lb=-np.inf, ub=b)

        # 3. Integrality: 1 indicates the variable must be an integer
        integrality = np.ones(n_vars)

        # 4. Variable Bounds:
        # By default, milp assumes x >= 0.
        # If your point can be anywhere in space (negative integers),
        # you MUST set bounds to +/- infinity.
        var_bounds = Bounds(lb=-np.inf, ub=np.inf)

        # 5. Solve
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=var_bounds)

        if res.success:
            # Rounding is safe here because constraints are satisfied within tolerance,
            # but pure integers are cleaner to return.
            return np.round(res.x).astype(int)
        else:
            return None

    @staticmethod
    def find_feasible_point(A: np.ndarray, b: np.ndarray) -> np.ndarray | None:
        """
        Determines if there is a solution x such that Ax <= b.

        Args:
            A (np.ndarray): The matrix of coefficients (m x n).
            b (np.ndarray): The vector of bounds (m).

        Returns:
            tuple: (is_feasible (bool), point (np.ndarray or None))
                   Returns a valid point 'x' if feasible, otherwise None.
        """
        # 1. Define the objective function (c).
        # We don't care about minimizing anything, just finding *any* point.
        # So we use a zero vector.
        n_vars = A.shape[1]
        c = np.zeros(n_vars)

        # 2. Call the solver.
        # 'highs' is the fastest open-source solver available in Scipy.
        # We must set bounds=(None, None) because by default linprog assumes x >= 0.
        res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None), method='highs')

        if res.success:
            return res.x
        else:
            # Check specific status codes if needed (2 = infeasible)
            return None

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

    # @staticmethod
    # def solve_polyhedron_fast(A: np.ndarray, b: np.ndarray, integer_only: bool = True) -> tuple[bool, np.ndarray | None]:
    #     """
    #     Checks if Ax <= b is feasible.
    #     optimized for speed by checking linear relaxation first.
    #
    #     Args:
    #         A: Coefficients matrix
    #         b: Bounds vector
    #         integer_only: If True, enforces integer solution.
    #
    #     Returns:
    #         (feasible, point)
    #     """
    #     n_vars = A.shape[1]
    #     c = np.zeros(n_vars)  # No objective, just feasibility
    #
    #     # --- Step 1: Linear Relaxation (The "Speed Filter") ---
    #     # We first check if a FLOATING POINT solution exists.
    #     # This is incredibly fast (P-Time). If this fails, integer solution is impossible.
    #     res_lp = linprog(c, A_ub=A, b_ub=b-1e-6, bounds=(None, None), method='highs')
    #
    #     if not res_lp.success:
    #         return False, None
    #
    #     # If user only wanted float, or if we got lucky and found integers naturally:
    #     if not integer_only:
    #         return True, res_lp.x
    #
    #     # Heuristic: Sometimes LP finds an integer solution by chance (e.g. at a clean vertex).
    #     # Check if the float solution is already close to integers.
    #     if np.allclose(res_lp.x, np.round(res_lp.x), atol=1e-5):
    #         return True, np.round(res_lp.x).astype(int)
    #
    #     # --- Step 2: Integer Solve (MILP) ---
    #     # Only runs if Step 1 passed but result wasn't integer.
    #     # This is the "heavy" lifting (NP-Hard).
    #
    #     # Convert constraints for milp format: -inf <= Ax <= b
    #     constraints = LinearConstraint(A, lb=-np.inf, ub=b)
    #     integrality = np.ones(n_vars)  # All variables must be integers
    #
    #     res_milp = milp(
    #         c=c,
    #         constraints=constraints,
    #         integrality=integrality,
    #         bounds=Bounds(-np.inf, np.inf),  # Allow negative integers
    #         options={"presolve": True}  # Critical for speed
    #     )
    #
    #     if res_milp.success:
    #         return True, np.round(res_milp.x).astype(int)
    #
    #     return False, None

    # @staticmethod
    # def get_signatures(A, b, points):
    #     """
    #     Vectorized calculation of sign patterns for many points.
    #     Returns set of unique tuples.
    #     """
    #     # points shape: (dim, n_samples)
    #     # A shape: (n_constraints, dim)
    #     # result shape: (n_constraints, n_samples)
    #
    #     # Check Ax > b (True if violated/flipped, False if standard)
    #     lhs = A @ points
    #     # Broadcasting b across columns
    #     is_flipped = lhs < b[:, np.newaxis]
    #
    #     # Convert columns to set of tuples
    #     # efficient matrix-to-set conversion
    #     return set(tuple(col) for col in is_flipped.T)

    # @staticmethod
    # def solve_bounded_feasibility(A, b, signs, box_limit):
    #     """
    #     Checks if a sign pattern exists strictly WITHIN the box [-box_limit, box_limit].
    #     """
    #     # 1. Setup Active Constraints based on signs
    #     # If sign is 0 (False): Ax <= b
    #     # If sign is 1 (True):  Ax >= b  -> -Ax <= -b
    #
    #     A_curr = A.copy()
    #     b_curr = b.copy()
    #
    #     # Flip rows where sign is True
    #     flip_idx = np.where(signs)[0]
    #     A_curr[flip_idx] *= 1
    #     b_curr[flip_idx] *= 1
    #
    #     n_vars = A.shape[1]
    #     c = np.zeros(n_vars)  # No objective needed
    #
    #     # 2. Apply the "Safety Box" Bounds
    #     # This prevents the solver from finding solutions at infinity
    #     # or outside your area of interest.
    #     bounds = (-box_limit, box_limit)
    #
    #     res = linprog(c, A_ub=A_curr, b_ub=b_curr, bounds=bounds, method='highs')
    #
    #     return res.success

    # @staticmethod
    # def find_regions_in_box(A, b, box_side=100, samples=100_000):
    #     """
    #     Identifies all feasible regions intersecting the hypercube centered at origin.
    #
    #     Args:
    #         A, b: Hyperplanes
    #         box_side: Length of the cube side (e.g. 100 means [-50, 50])
    #         samples: Number of random points to test
    #     """
    #     limit = box_side / 2.0
    #     dim = A.shape[1]
    #     n_planes = A.shape[0]
    #
    #     print(f"--- Starting Search in {dim}D Box [{-limit}, {limit}] ---")
    #
    #     # --- Phase 1: Uniform "Flood" Sampling ---
    #     # Much safer than Gaussian because we cover corners equally
    #     print(f"Phase 1: Sampling {samples} points...")
    #
    #     random_points = np.random.uniform(low=-limit, high=limit, size=(dim, samples))
    #     found_regions = Shard.get_signatures(A, b, random_points)
    #
    #     print(f"  > Found {len(found_regions)} unique regions via sampling.")
    #
    #     # --- Phase 2: Bounded Crawler ---
    #     # We use the regions found by sampling as our starting "seeds".
    #     # We explore their neighbors to find any tiny slivers sampling might have missed.
    #
    #     queue = list(found_regions)
    #     checked = set(found_regions)  # Avoid re-checking known ones
    #     valid = set(found_regions)  # Final list of valid ones
    #
    #     print("Phase 2: Crawling for missing neighbors...")
    #
    #     while queue:
    #         current_sign = queue.pop(0)
    #
    #         for i in range(n_planes):
    #             # Create neighbor signature (flip i-th bit)
    #             neigh_lst = list(current_sign)
    #             neigh_lst[i] = not neigh_lst[i]
    #             neighbor = tuple(neigh_lst)
    #
    #             if neighbor not in checked:
    #                 checked.add(neighbor)
    #
    #                 # Check feasibility ONLY within the box
    #                 is_feasible = Shard.solve_bounded_feasibility(A, b, neighbor, limit)
    #
    #                 if is_feasible:
    #                     valid.add(neighbor)
    #                     queue.append(neighbor)  # Continue crawling from this new region
    #
    #     print(f"--- Finished. Total valid regions: {len(valid)} ---")
    #     return list(valid)

    # @cached_property
    # def start_coord(self) -> Position:
    #     def find_integer_solution(A, b):
    #         """
    #         Checks for integer solution to Ax < b.
    #         Returns the solution x if found, else None.
    #         """
    #         m, n = A.shape
    #
    #         # 1. Handle the strict inequality (Ax < b)
    #         # If data is purely integer, use offset = 1.0
    #         # If data is float, use a small epsilon, e.g., offset = 1e-6
    #         offset = 1e-6
    #         b_upper = b - offset
    #
    #         # 2. Define Constraints: -inf <= Ax <= b_upper
    #         # We use -np.inf for the lower bound effectively making it a one-sided inequality
    #         constraints = LinearConstraint(A, -np.inf, b_upper)
    #
    #         # 3. Define Integrality: 1 means integer, 0 means continuous
    #         integrality = np.ones(n)
    #
    #         # 4. Define Bounds on X: default is (0, inf), we want (-inf, inf)
    #         # Note: Solvers work faster with tighter bounds, but this works generally.
    #         bounds = Bounds(-np.inf, np.inf)
    #
    #         # 5. Objective: We only care about feasibility, so we minimize 0*x
    #         c = np.zeros(n)
    #
    #         res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    #
    #         if res.success:
    #             # Rounding is safe because the solver guarantees integer feasibility within tolerance
    #             return np.round(res.x).astype(int)
    #         else:
    #             return None
    #
    #     # res = self.__find_integer_point_milp(
    #     #     self.A, self.b_shifted,
    #     #     xmin=[-analysis_config.VALIDATION_BOUND_BOX_DIM] * self.dim,
    #     #     xmax=[analysis_config.VALIDATION_BOUND_BOX_DIM] * self.dim
    #     # )
    #     # if self.find_feasible_point(self.A, self.b) is None:
    #     #     return None
    #     # res = self.find_integer_feasible_point(self.A, self.b)
    #     # res = self.solve_polyhedron_fast(self.A, self.b_shifted, True)[1]
    #     # res = find_integer_solution(self.A, self.b_shifted)
    #     if self.start_coord is not None:
    #         return self.start_coord
    #     res = self.find_integer_feasible_point(self.A, self.b_shifted - 1e-4)
    #     if res is None:
    #         return None
    #     return Position({sym: v for sym, v in zip(self.symbols, np.int64(res).tolist())}) + Position({sym: sp.Rational(v) for sym, v in zip(self.symbols, self.shift.tolist())})

    # @cached_property
    # def is_valid(self):
    #     return self.start_coord is not None

    def __repr__(self):
        return f'A={self.A}\nb={self.b}'

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

    # 2. Cone Check (Ax < b)
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
def chrr_walker(A, A_cols, b, R_sq, start_point, n_desired, thinning, buf_out, max_steps):
    """
    A: (m, d)
    A_cols: (d, m) - Transposed A for fast column access
    """
    ERROR_BOUND = 1e-8

    m, d = A.shape

    # Current State
    x = start_point.copy()
    current_Ax = np.dot(A, x)

    found = 0
    steps = 0

    # Temp buffer for integer check
    temp_int = np.zeros(d, dtype=np.int64)

    while found < n_desired and steps < max_steps:
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
                    if dot > b[row] + ERROR_BOUND:
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
    return found, steps


class CHRRSampler:
    def __init__(self, A, b, R, thinning=3, start=None):
        self.A = np.ascontiguousarray(A, dtype=np.float64)
        # We pre-transpose A for faster column access in the walker
        self.A_cols = np.ascontiguousarray(A.T, dtype=np.float64)
        self.b = np.ascontiguousarray(b, dtype=np.float64)
        self.R = float(R)
        self.R_sq = self.R**2
        self.thinning = thinning
        self.start = start

    def find_start_point(self):
        """Finds a valid continuous point inside to start the chain."""
        # Simple Rejection Sampling Attempt (usually finds one instantly)
        # If the cone is SO thin this fails, we need LP, but let's assume this works.
        if self.start is not None:
            return self.start
        d = self.A.shape[1]
        for _ in range(10000):
            # Sample in small box around origin or uniform in R
            cand = np.random.uniform(-self.R/10, self.R/10, size=d)
            if np.linalg.norm(cand) > self.R:
                continue
            if np.all(self.A @ cand < self.b):
                return cand
        raise RuntimeError("Could not find starting point for CHRR. Cone is too thin or closed.")

    def sample(self, n_samples):
        t0 = time.time()
        try:
            start_pt = self.find_start_point()
        except:
            start_pt = None

        max_steps_per_round = max(2000 * n_samples * self.thinning, 100_000)

        buf = np.zeros((n_samples, self.A.shape[1]), dtype=np.int64)

        retries = 0
        total_found = 0
        while total_found < n_samples and retries < 5:
            # 1. Find valid start point for CURRENT Radius
            start_pt = self.find_start_point() if start_pt is None else start_pt
            needed = n_samples #- total_found

            found_now, _ = chrr_walker(
                self.A, self.A_cols, self.b, self.R_sq,
                start_pt, needed, self.thinning,
                buf[total_found:], max_steps_per_round
            )

            total_found += found_now

            # 3. Check Success
            if total_found < n_samples:
                # We failed to find enough points. The Radius is likely too small.
                # Expand Radius by 50%
                old_R = self.R
                self.R *= 1.5
                self.R_sq = self.R ** 2
                retries += 1
        #
        # chrr_walker(self.A, self.A_cols, self.b, self.R_sq, start_pt, n_samples, self.thinning, buf, max_steps)
        if retries == 5:
            Logger(
                f'Number of trajectories is too small, try increasing radius or number of retries'
                f' (THIS COULD BE A BUG - CONTACT DEV)', Logger.Levels.warning
            ).log(msg_prefix='\n')
        unique_set = set(tuple(x) for x in buf)
        final_arr = np.array(list(unique_set))
        return final_arr, time.time() - t0


if __name__ == '__main__':
    a = np.array([[1, 2], [3, 4]])
    b = np.array([1, 1])
    x, y = sp.symbols('x y')
    shard = Shard(a, b, Position({x: 0.5, y: 0.5}), [x, y])
    print(shard.b_shifted)
