"""
Extractor is responsible for shard creation
"""
import itertools
import os.path
from collections import defaultdict

from dreamer.utils.schemes.extraction_scheme import ExtractionScheme, ExtractionModScheme
from dreamer.utils.types import *

from dreamer.configs import sys_config, extraction_config

from dreamer.configs import analysis_config
from concurrent.futures import ProcessPoolExecutor
from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.extraction.shard import Shard
from dreamer.utils.logger import Logger
from dreamer.utils.constants.constant import Constant
from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.storage.exporter import Exporter
from dreamer.utils.storage.formats import Formats
from ramanujantools.cmf.d_finite import theta


from tqdm import tqdm
import numpy as np
from scipy.optimize import linprog
from collections import deque


class ShardExtractorMod(ExtractionModScheme):
    def __init__(self, cmf_data: Dict[Constant, List[ShiftCMF]]):
        super().__init__(
            cmf_data,
            name=self.__class__.__name__,
            desc='Shard extractor module',
            version='v0.0.1'
        )

    def execute(self) -> Dict[Constant, List[Searchable]]:
        all_shards = defaultdict(list)

        consts_itr = iter(list(self.cmf_data.keys()))
        Logger.sleep(0.5)
        for const, cmf_list in tqdm(self.cmf_data.items(), desc=f'Extracting shards for "{next(consts_itr).name}"',
                                    **sys_config.TQDM_CONFIG):
            with Exporter.export_stream(
                    os.path.join(extraction_config.PATH_TO_SEARCHABLES, const.name),
                    exists_ok=True, clean_exists=True, fmt=Formats.PICKLE
            ) as export_stream:
                ind_itr = iter(range(len(cmf_list)))
                Logger.sleep(0.5)
                for cmf_shift in tqdm(cmf_list, desc=f'Computing shards for CMF no. {next(ind_itr) + 1}',
                                      **sys_config.TQDM_CONFIG):
                    extractor = ShardExtractor(const, cmf_shift.cmf, cmf_shift.shift)
                    shards = extractor.extract_searchables()
                    all_shards[const] += shards
                    export_stream(shards)
                Logger.sleep(0.5)
        return all_shards


class ShardExtractor(ExtractionScheme):
    def __init__(self, const: Constant, cmf: CMF, shift: Position):
        super().__init__(const, cmf, shift)
        self.pool = ProcessPoolExecutor() if analysis_config.PARALLEL_SHARD_VALIDATION else None

    @property
    def symbols(self) -> List[sp.Symbol]:
        return list(self.cmf.matrices.keys())

    @staticmethod
    def extract_matrix_hps(mat, shift: Position, symbols: List[sp.Symbol]) -> List[Hyperplane]:
        """
        Extracts HPs from the matrix with respect to shift (ignore irrelevant HPs)
        :param mat: Matrix to extract HPs from
        :param shift: shift for HPs
        :param symbols: symbols used by the CMF
        :return: set of hyperplanes filtered and shifted
        """
        # Zero division solutions
        l = set()

        import sympy as sp

        for v in mat.iter_values():
            if (den := v.as_numer_denom()[1]) == 1:
                continue

            solutions = {(sym, sol) for sym in den.free_symbols for sol in sp.solve(sp.simplify(den), sym)}
            for lhs, rhs in solutions:
                l.add(Hyperplane(lhs - rhs, symbols))

        # Zero det solutions
        solutions = [tuple(*sol.items()) for sol in sp.solve(sp.simplify(mat.det()))]
        hps = l.union({Hyperplane(lhs - rhs, symbols) for lhs, rhs in solutions})

        filtered_hps = []
        for hp in hps:
            shifted = hp.apply_shift(shift)
            if shifted.is_in_integer_shift():
                filtered_hps.append(hp)
        return filtered_hps

    def extract_cmf_hps(self) -> Set[Hyperplane]:
        """
        Compute the hyperplanes of the CMF a shifted trajectory could encounter
        :return: two sets - the filtered hyperplanes and the shifted hyperplanes
        """
        filtered_hps = set()
        symbols = list(self.cmf.matrices.keys())
        for s in symbols:
            zeros = sp.solve(determinant_from_char_poly(self.cmf.p, self.cmf.q, self.cmf.z, s))
            zeros = [Hyperplane(lhs - rhs, symbols) for solution in zeros for lhs, rhs in solution.items()]
            filtered_hps.update(set(zeros))

            poles = set()
            for v in self.cmf.matrices[s].iter_values():
                if (den := v.as_numer_denom()[1]) == 1:
                    continue

                solutions = {(sym, sol) for sym in den.free_symbols for sol in sp.solve(sp.simplify(den), sym)}
                for lhs, rhs in solutions:
                    poles.add(Hyperplane(lhs - rhs, symbols))
            filtered_hps.update(poles)

        return filtered_hps

    def extract_searchables(self) -> List[Shard]:
        """
        Extracts the shards from the CMF
        :return: The set of shards matching the CMF
        """
        hps = self.extract_cmf_hps()
        shards = []

        ws = []
        bs = []
        for hp in hps:
            w, b = hp.vectors
            ws.append(w)
            bs.append(b)

        Logger(
            f'Found {len(hps)} hyperplanes',
            level=Logger.Levels.info
        ).log(msg_prefix='\n')

        symbols = list(hps)[0].symbols
        points = [tuple(coord + shift for coord, shift in zip(p, self.shift.values())) for p in list(itertools.product(tuple(list(range(-2, 3))), repeat=len(symbols)))]
        shard_encodings = dict()
        for p in tqdm(points, desc='Checking shard encodings ...', **sys_config.TQDM_CONFIG):
            enc = []
            point_dict = {sym: coord for sym, coord in zip(symbols, p)}
            for hp in hps:
                res = hp.expr.subs(point_dict)
                if res == 0:
                    break
                if res > 0:
                    res = 1
                elif res < 0:
                    res = -1
                enc.append(res)
            if len(enc) == len(hps):
                shard_encodings[tuple(enc)] = Position(point_dict)

        for enc in tqdm(shard_encodings.keys(), desc='Creating shard objects', **sys_config.TQDM_CONFIG):
            A, b, syms = Shard.generate_matrices(list(hps), enc)
            shards.append(Shard(self.cmf, self.const, A, b, self.shift, syms, shard_encodings[enc]))

        Logger(
            f'Found {len(shard_encodings)} shards',
            level=Logger.Levels.info
        ).log(msg_prefix='\n')
        return shards


def determinant_from_char_poly(p, q, z, axis: sp.Symbol):
    # substitute in differential equation & extract free coeff of normalized characteristic poly
    # if y axis then increment the parameter
    is_y_shift = True if axis.name.startswith("y") else False
    coeff = axis - 1 if axis.name.startswith("y") else axis
    S = sp.symbols("S")  # note that for a y shift, we're calculating the char poly for S^{-1}
    theta_subs = coeff * S - coeff

    differential_equation = pFq.differential_equation(p, q, z).subs({z: z})

    char_poly_for_S = sp.monic(pFq.differential_equation(p, q, z).subs({theta: theta_subs}),
                               S)  # how does this handle z eval?
    free_coeff = char_poly_for_S.coeff_monomial(1)  # can also just subs

    matrix_dim = char_poly_for_S.degree()

    if is_y_shift:
        return sp.factor((((-1) ** matrix_dim) / free_coeff).subs({axis: axis + 1}))
    else:
        return sp.factor((-1) ** matrix_dim * free_coeff)


class HyperplaneArrangement:
    def __init__(self, hyperplanes, biases):
        self.W = np.array(hyperplanes, dtype=np.float64)
        self.B = np.array(biases, dtype=np.float64)
        self.N, self.d = self.W.shape
        self.R = None

    def estimate_radius(self, n_samples=50000, safety_factor=10.0):
        """
        Estimates a 'safe' bounding radius.
        safety_factor is high (10.0) to ensure we don't cut off long cones.
        """
        # Generate random indices
        idx_list = np.random.randint(0, self.N, size=(n_samples, self.d))
        W_batch = self.W[idx_list]
        B_batch = -self.B[idx_list]

        try:
            vertices = np.linalg.solve(W_batch, B_batch)
            norms = np.linalg.norm(vertices, axis=1)
            valid_norms = norms[np.isfinite(norms)]

            if len(valid_norms) == 0:
                max_norm = 100.0
            else:
                max_norm = np.max(valid_norms)
        except:
            max_norm = 100.0

        self.R = max_norm * safety_factor
        print(f"Radius set to: {self.R:.2f}")
        return self.R

    def find_neighbors_exact(self, signs, W_norm, B_norm):
        """
        Determines neighbors by checking if the 'face' created by each
        hyperplane is feasible within the current cone.

        Solves 1 LP per hyperplane.
        """
        neighbors = []

        # Pre-compute the cone constraints: Ax <= b
        # -s * w * x <= s * b
        A_cone = -signs[:, None] * W_norm
        b_cone = signs * B_norm

        # Reuse bounds
        bounds = [(-self.R, self.R)] * self.d

        for i in range(self.N):
            # Optimization: If the constraint is already violated or super loose
            # in the center, we might skip, but for Robustness, we run the LP.

            # We want to check if there exists an x such that:
            # 1. x is in the cone (A_cone @ x <= b_cone)
            # 2. x is ON the wall i (w_i @ x + b_i = 0)

            # To use linprog (inequality only), we replace equality w_i.x + b_i = 0
            # with two inequalities: w_i.x + b_i <= 0  AND  -w_i.x - b_i <= 0

            # However, simpler trick:
            # Just set the i-th constraint in A_cone to be Equality?
            # Scipy separates A_eq and A_ub.

            # A_ub: All cone constraints EXCLUDING i
            # A_eq: The i-th constraint

            # Mask for all indices except i
            mask = np.arange(self.N) != i

            A_ub = A_cone[mask]
            b_ub = b_cone[mask]

            A_eq = self.W[i:i + 1]  # Use original W for equality check to be precise
            b_eq = -self.B[i:i + 1]

            # Feasibility Check (Minimize 0)
            c = np.zeros(self.d)

            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds, method='highs')

            if res.success:
                neighbors.append(i)

        return neighbors

    def find_all_cones(self):
        if self.R is None: self.estimate_radius()

        # Normalize
        W_norms = np.linalg.norm(self.W, axis=1)
        W_norms[W_norms < 1e-9] = 1.0
        W_norm = self.W / W_norms[:, None]
        B_norm = self.B / W_norms

        # Seed Finding
        for _ in range(100):
            seed = np.random.uniform(-self.R / 100, self.R / 100, self.d)
            if np.linalg.norm(seed) < self.R: break

        initial_signs = np.sign(self.W @ seed + self.B).astype(int)
        initial_signs[initial_signs == 0] = 1

        queue = deque([tuple(initial_signs)])
        visited = {tuple(initial_signs)}
        results = []

        print(f"Starting Exact BFS (N={self.N}, d={self.d})...")

        while queue:
            current_s_tuple = queue.popleft()
            s = np.array(current_s_tuple)

            # 1. Store the cone (We assume it's valid if we queued it)
            # Optional: You can solve for the Center here if you need it for sampling later,
            # but for pure enumeration, we don't strictly need it.
            results.append(current_s_tuple)

            # 2. Find Neighbors EXACTLY
            active_walls = self.find_neighbors_exact(s, W_norm, B_norm)

            for idx in active_walls:
                new_s = list(current_s_tuple)
                new_s[idx] *= -1
                new_s_tuple = tuple(new_s)

                if new_s_tuple not in visited:
                    visited.add(new_s_tuple)
                    queue.append(new_s_tuple)

        return results


if __name__ == '__main__':
    x0, x1, y0, n = sp.symbols('x0 x1 y0 n')
    from ramanujantools.cmf import pFq
    # This is pi 2F1 CMF
    pi = pFq(2, 1, sp.Rational(1, 2))

    shift = Position({x0: sp.Rational(1, 2), x1: sp.Rational(1,2), y0: sp.Rational(1,2)})
    # pprint(ShardExtractor('pi', pi, shift).extract_cmf_hps())
    # ppt = ShardExtractor('pi', pi, shift).extract_shards()
    # pprint(len(ppt))
    shifted = Hyperplane(x0+1, [x0, x1, y0]).apply_shift(shift)
    print(shifted.expr)
    print(shifted.is_in_integer_shift())
