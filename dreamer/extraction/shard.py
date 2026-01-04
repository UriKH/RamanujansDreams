from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.types import *
from dreamer.utils.constants.constant import Constant
from .sampler_chrr import CHRRSampler
from .sampler_sphere import PrimitiveSphereSampler

from scipy.special import gamma, zeta
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
import numpy as np


class Shard(Searchable):
    def __init__(self,
                 cmf: CMF,
                 constant: Constant,
                 A: np.ndarray | None,
                 b: np.ndarray | None,
                 shift: Position,
                 symbols: List[sp.Symbol],
                 interior_point: Optional[Position] = None
                 ):
        """
        :param cmf: The CMF this shard is a part of
        :param constant: The constant to search for in the shard
        :param A: Matrix A defining the linear terms in the inequalities
            (if None, then the shard will be the whole space)
        :param b: Vector b defining the free terms in the inequalities
            (if None, then the shard will be the whole space)
        :param shift: The shift in start points required
        :param symbols: Symbols used by the CMF which this shard is part of
        :param interior_point: A point within the shard
        """
        super().__init__(cmf, constant, shift)
        self.A = A
        self.b = b
        self.symbols = symbols
        self.shift = np.array([shift[sym] for sym in self.symbols])
        self.start_coord = interior_point
        self.is_whole_space = self.A is None or self.b is None

    def in_space(self, point: Position) -> bool:
        """
        Checks if a point is inside the shard.
        :param point: A point to check if it is inside the shard
        :return: True if A @ point < b else False
        """
        if self.is_whole_space:
            return True
        point = np.array(point.sorted().values())
        return np.all(self.A @ point < self.b)

    def get_interior_point(self) -> Position:
        """
        :return: A point inside the shard
        """
        if not self.start_coord:
            return Position({s: sp.Integer(0) for s in self.symbols})
        return Position({sym: v for v, sym in zip(self.start_coord.values(), self.symbols)})

    def sample_trajectories(self, n_samples, *, strict: Optional[bool] = False) -> Set[Position]:
        """
        Sample trajectories from the shard.
        :param n_samples: Number of samples to generate
        :param strict: True -> compute as n_samples, else compute n_samples * fraction.
        (fraction of the cone is taking from the sphere)
        :return: A set of sampled trajectories
        """
        def _estimate_cone_fraction(A, n_trials=5000) -> float:
            """
            Helper to estimate what % of the sphere is covered by the cone.
            """
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

        if self.is_whole_space:
            samples = PrimitiveSphereSampler(len(self.symbols)).sample(n_samples)
            return {
                Position({sym: sp.sympify(v) for v, sym in zip(p, self.symbols)})
                for p in samples
            }

        fraction = _estimate_cone_fraction(self.A) * 1.05   # always assume undersampling
        if strict:
            # We need n_samples INSIDE the cone.
            n_target_safe = int((n_samples / fraction) * 1.2)   # go on the safe side assuming bad fraction estimation
            R = self.compute_ball_radius(len(self.symbols), n_target_safe)
            target_yield = n_samples
        else:
            # We treat n_samples as the "Density" of the full sphere.
            R = self.compute_ball_radius(len(self.symbols), n_samples)
            target_yield = int(n_samples * fraction)

            # Edge case: If cone is tiny, ensure we at least look for more than 1
            if target_yield < 1:
                target_yield = 10

        sampler = CHRRSampler(
            self.A, np.zeros_like(self.b), R=np.ceil(R + 0.5), thinning=5,
            start=np.array(list(self.get_interior_point().values()), dtype=np.float64)
        )
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
        :param d: Number of dimensions.
        :param n_samples: Target number of points.
        :return: The estimated Radius R.
        """
        vol_unit_ball = (np.pi ** (d / 2.)) / gamma(d / 2. + 1.)
        density = 1. / zeta(d) if d > 1 else 1.
        term = n_samples / (vol_unit_ball * density)
        R = term ** (1. / d)
        return R

    # TODO: unused method - might be relevant
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

    # TODO: unused method - might be relevant
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
        """
        Generate the matrix A and vector b corresponding to the given hyperplanes which represent a shard
        with a specific encoding.
        :param hyperplanes: The list of hyperplanes that represent the shard.
        :param above_below_indicator: The indicator vector that indicates whether the shard is below or above
        the hyperplanes.
        :return: (A, b) where A is a matrix with rows as the linear term coefficients of the hyperplanes
        and b is the free terms vector.
        """
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

    def __repr__(self):
        return f'A={self.A}\nb={self.b}'


if __name__ == '__main__':
    a = np.array([[1, 2], [3, 4]])
    b = np.array([1, 1])
    x, y = sp.symbols('x y')
    shard = Shard(a, b, Position({x: 0.5, y: 0.5}), [x, y])
    print(shard.b_shifted)
