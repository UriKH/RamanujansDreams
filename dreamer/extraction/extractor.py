"""
Extractor is responsible for shard creation
"""
import itertools
import os.path
import time
from collections import defaultdict
import sympy as sp
from functools import partial
import random

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
        def get_numerator_matrix(matrix):
            """
            Step 1: Clears denominators from a matrix of rational functions.
            Returns a matrix of polynomials.
            """
            rows, cols = matrix.shape
            matrix_poly = sp.zeros(rows, cols)

            for i in range(rows):
                row = matrix.row(i)
                # 1. Gather denominators
                denoms = [sp.fraction(entry)[1] for entry in row]
                # 2. Find LCM for the row
                row_lcm = sp.lcm(denoms)
                # 3. Multiply row by LCM to get purely polynomial entries
                for j in range(cols):
                    # simplify is needed to cancel the denominator with the LCM
                    matrix_poly[i, j] = sp.simplify(row[j] * row_lcm)

            return matrix_poly

        def triangularize_and_find_hyperplanes(matrix):
            """
            Step 2: Performs Gaussian Elimination while aggressively factoring
            intermediate terms to isolate hyperplane solutions.
            """
            # Work on a copy to avoid modifying original
            M = matrix.copy()
            rows, cols = M.shape

            # print(f"--- Starting Factored Gaussian Elimination ({rows}x{cols}) ---")

            # We will collect factors found on the diagonal
            potential_factors = []

            for k in range(rows - 1):
                pivot = M[k, k]

                # A. Swap if pivot is identically zero
                if pivot == 0:
                    for i in range(k + 1, rows):
                        if M[i, k] != 0:
                            M.row_swap(k, i)
                            pivot = M[k, k]
                            # print(f"   [Col {k}] Swapped row {k} with {i}")
                            break
                    else:
                        # If column is all zero, determinant is 0.
                        # This means 0 is a factor (always singular).
                        return [sp.Integer(0)]

                # B. Store the pivot as a potential factor
                potential_factors.append(pivot)

                # C. Elimination Loop
                for i in range(k + 1, rows):
                    target_val = M[i, k]

                    if target_val == 0:
                        continue

                    # We use the cross-multiplication method to stay in Polynomial land:
                    # Row_i = (pivot * Row_i) - (target * Row_k)
                    # This avoids fractions entirely.

                    for j in range(k, cols):
                        # Calculate the unsimplified new entry
                        new_val = (pivot * M[i, j]) - (target_val * M[k, j])

                        # CRITICAL: Factor immediately.
                        # This stops the expression from exploding and reveals the hyperplanes.
                        M[i, j] = sp.factor(new_val)

                # print(f"   [Col {k}] Elimination done. Matrix entries factored.")

            # Append the final element (bottom right)
            potential_factors.append(M[rows - 1, cols - 1])

            return potential_factors

        def extract_unique_planes_v3(factors):
            unique_planes = set()

            for term in factors:
                # 1. Factor to separate products
                factored_term = sp.factor(term)

                # 2. Get base factors (stripping exponents)
                powers_map = factored_term.as_powers_dict()

                for base, exponent in powers_map.items():
                    if not base.free_symbols:
                        continue

                    # --- THE MAGIC FIX ---
                    # as_content_primitive() separates the scalar (1/5) from the equation (x+y+5)
                    # We discard the content (scalar) and keep the clean equation.
                    content, integer_plane = base.as_content_primitive()

                    # Optional: Ensure the first coefficient is positive for consistency
                    # (e.g., turns -x - y - 5 = 0 into x + y + 5 = 0)
                    # We grab the coeff of the first variable we find
                    first_symbol = list(integer_plane.free_symbols)[0]
                    if sp.Poly(integer_plane).total_degree() > 1:
                        continue
                    # if integer_plane.coeff(first_symbol) < 0:
                    #     integer_plane = -integer_plane

                    unique_planes.add(integer_plane)

            return list(unique_planes)

        # ==================================== SECOND METHOD ===================================
        def get_numerator_matrix(matrix):
            """Clears denominators to get a polynomial matrix."""
            rows, cols = matrix.shape
            matrix_poly = sp.zeros(rows, cols)
            for i in range(rows):
                row = matrix.row(i)
                denoms = [sp.fraction(entry)[1] for entry in row]
                row_lcm = sp.lcm(denoms)
                for j in range(cols):
                    matrix_poly[i, j] = sp.simplify(row[j] * row_lcm)
            return matrix_poly

        def get_candidates_dirty(matrix):
            """
            Runs a fast, 'dirty' elimination that collects potential factors.
            It intentionally collects 'too much' to ensure we don't miss anything.
            """
            M = matrix.copy()
            rows, cols = M.shape
            candidates = []

            for k in range(rows - 1):
                pivot = M[k, k]

                # 1. Pivot Strategy: Just swap if 0
                if pivot == 0:
                    for i in range(k + 1, rows):
                        if M[i, k] != 0:
                            M.row_swap(k, i)
                            pivot = M[k, k]
                            break
                    else:
                        return [sp.Integer(0)]  # Zero determinant

                # CRITICAL: The pivot itself might be a solution!
                candidates.append(pivot)

                for i in range(k + 1, rows):
                    if M[i, k] == 0: continue

                    # Cross-multiply
                    factor_val = M[i, k]
                    for j in range(k, cols):
                        # We calculate the new value
                        val = (pivot * M[i, j]) - (factor_val * M[k, j])
                        # We factor it to keep it clean
                        M[i, j] = sp.factor(val)

                    # Collect the diagonal entry as we create it
                    if i == j:  # specific check for diagonal
                        candidates.append(M[i, j])

            # Add the final element
            candidates.append(M[rows - 1, cols - 1])
            return candidates

        def verify_hyperplane(plane_eq, original_matrix, variables, trials=2):
            """
            Checks if a candidate hyperplane is a TRUE solution.
            Method: Monte Carlo Substitution.
            1. Pick random integers for all variables EXCEPT one.
            2. Solve for the last variable to force the point onto the hyperplane.
            3. Plug point into original matrix.
            4. Check if Det == 0.
            """
            # 0. Trivial check
            if plane_eq == 0: return True
            if plane_eq.is_Number: return False

            # 1. Pick a target variable to solve for (e.g., z)
            # We need a variable that is actually in the plane equation
            plane_vars = list(plane_eq.free_symbols)
            if not plane_vars: return False
            target_var = plane_vars[0]  # Just pick the first one
            other_vars = [v for v in variables if v != target_var]

            for _ in range(trials):
                # 2. Generate random point P
                # Map other variables to random integers
                subs_dict = {v: random.randint(1, 10) for v in other_vars}

                # Calculate what the target variable MUST be to stay on the plane
                # plane_eq(x,y,z) = 0  =>  solve for target_var
                try:
                    # Solve eq = 0 for target_var numerically
                    # We treat the random integers as constants
                    eq_subbed = plane_eq.subs(subs_dict)
                    target_val_solutions = sp.solve(eq_subbed, target_var)

                    if not target_val_solutions: return False  # No solution? Bad candidate.
                    target_val = target_val_solutions[0]

                    # Update dict with the calculated coordinate
                    subs_dict[target_var] = target_val

                except:
                    return False

                # 3. Plug P into Original Matrix
                # We assume original matrix might have rational entries, so we compute det
                # We can use domain='QQ' for speed since we substituted numbers
                try:
                    M_val = original_matrix.subs(subs_dict)
                    # Use bareiss for fast numerical determinant
                    det_val = M_val.det(method='bareiss')

                    # 4. Check if Det is ZERO
                    # We use simplify to handle potentially complex cancellations
                    if sp.simplify(det_val) != 0:
                        return False
                except:
                    return False

            return True

        # def clean_and_verify_solutions(matrix, variables):
        #     """
        #     Main Driver Function.
        #     """
        #     M_poly = get_numerator_matrix(matrix)
        #     raw_candidates = get_candidates_dirty(M_poly)
        #
        #     unique_candidates = set()
        #     verified_planes = set()
        #
        #     # Clean raw candidates (strip powers, contents)
        #     for item in raw_candidates:
        #         factored = sp.factor(item)
        #         if isinstance(factored, sp.Mul):
        #             factors = factored.args
        #         else:
        #             factors = [factored]
        #
        #         for f in factors:
        #             powers = f.as_powers_dict()
        #             for base, exp in powers.items():
        #                 if not base.free_symbols:
        #                     continue
        #
        #                 content, clean = base.as_content_primitive()
        #
        #                 # 3. FAST Sign Normalization
        #                 # We want to turn (-x - y) into (x + y)
        #                 # But we must avoid crashing on (x*y + z)
        #
        #                 # Get a deterministic 'first' symbol (e.g., 'x')
        #                 # (Sorting a list of 5 symbols is instant)
        #                 free_syms = sorted(list(clean.free_symbols), key=lambda s: s.name)
        #                 first_sym = free_syms[0]
        #
        #                 # Get the coefficient of this symbol
        #                 # e.g., in (-2x + y), coeff of x is -2.
        #                 coeff = clean.coeff(first_sym)
        #
        #                 # CRITICAL FIX: Only compare if it's a pure Number!
        #                 # If coeff is 'y', we skip this step.
        #                 if coeff.is_number and coeff < 0:
        #                     clean = -clean
        #
        #                 unique_candidates.add(clean)
        #
        #                 # # Strip scalar content (e.g. 5x -> x)
        #                 # content, clean = base.as_content_primitive()
        #                 # # Normalize sign
        #                 # first_sym = list(clean.free_symbols)[0]
        #                 # if sp.total_degree(clean) > 1:
        #                 #     continue
        #                 # if clean.coeff(first_sym) < 0:
        #                 #     clean = -clean
        #                 # unique_candidates.add(clean)
        #
        #     # Verify each candidate
        #     for cand in unique_candidates:
        #         if verify_hyperplane(cand, matrix, variables):
        #             verified_planes.add(cand)
        #
        #     return list(verified_planes)

        import sympy as sp
        import random

        def clean_and_verify_safe(matrix, variables, max_expected_degree=3):
            """
            Stabilized version:
            1. Deterministic (Fixed Seed)
            2. Prevents hangs (Degree Guard)
            3. Filters artifacts robustly
            """
            # FIX 1: Set Random Seed for reproducibility
            # This ensures "different runs" always give the SAME result.
            random.seed(42)

            print("Step 1: Pre-processing Matrix...")
            M_poly = get_numerator_matrix(matrix)

            print("Step 2: Harvesting Candidates (Dirty Method)...")
            raw_candidates = get_candidates_dirty(M_poly)
            print(f"   -> Collected {len(raw_candidates)} raw candidates.")

            unique_candidates = set()
            verified_planes = set()

            print("Step 3: Cleaning Candidates (With Degree Guard)...")
            for item in raw_candidates:
                # FIX 2: THE DEGREE GUARD
                # Before doing expensive operations, check complexity.
                # If a polynomial is massive, it's definitely an artifact of the algorithm.
                try:
                    # Using total_degree is fast. If it fails or is huge, skip.
                    deg = sp.total_degree(item)
                    if deg > max_expected_degree:
                        # print(f"      [Skipping garbage of degree {deg}]")
                        continue
                except:
                    continue

                # Now it is safe to factor
                try:
                    factored = sp.factor(item)
                except:
                    continue

                if isinstance(factored, sp.Mul):
                    factors = factored.args
                else:
                    factors = [factored]

                for f in factors:
                    powers = f.as_powers_dict()
                    for base, exp in powers.items():
                        if not base.free_symbols: continue

                        # Cleanup logic
                        content, clean = base.as_content_primitive()

                        # Fast sign normalization
                        free_syms = sorted(list(clean.free_symbols), key=lambda s: s.name)
                        if not free_syms: continue
                        first_sym = free_syms[0]
                        coeff = clean.coeff(first_sym)

                        if coeff.is_number and coeff < 0:
                            clean = -clean

                        unique_candidates.add(clean)

            print(f"   -> Reduced to {len(unique_candidates)} reasonable candidates.")

            # Verify each candidate
            print("Step 4: Numerically Verifying...")
            for cand in unique_candidates:
                # Double check degree just in case
                if sp.total_degree(cand) > max_expected_degree:
                    continue

                # FIX 3: Increased Trials and Range for better accuracy
                if verify_hyperplane_robust(cand, matrix, variables):
                    print(f"   [VALID] {cand} = 0")
                    verified_planes.add(cand)

            return list(verified_planes)

        def verify_hyperplane_robust(plane_eq, original_matrix, variables):
            """
            More robust verification with wider random range to prevent accidental zeros.
            """
            if plane_eq == 0: return True
            if plane_eq.is_Number: return False

            plane_vars = list(plane_eq.free_symbols)
            if not plane_vars: return False
            target_var = plane_vars[0]
            other_vars = [v for v in variables if v != target_var]

            # Run 3 trials to be sure
            for _ in range(3):
                # Use wider range (-50 to 50) to avoid "accidental" small integer zeros
                subs_dict = {v: random.randint(-50, 50) for v in other_vars}

                try:
                    eq_subbed = plane_eq.subs(subs_dict)
                    # Solve for target
                    target_sols = sp.solve(eq_subbed, target_var)
                    if not target_sols: return False

                    # Pick the first solution
                    target_val = target_sols[0]
                    subs_dict[target_var] = target_val

                    # Check determinant
                    # We use bareiss on the substituted matrix (which is now all numbers)
                    M_val = original_matrix.subs(subs_dict)
                    det_val = M_val.det(method='bareiss')

                    if sp.simplify(det_val) != 0:
                        return False
                except:
                    return False

            return True

        # M_poly = get_numerator_matrix(mat)
        # raw_factors = triangularize_and_find_hyperplanes(M_poly)
        # solutions = extract_unique_planes_v3(raw_factors)
        # l3 = set()
        # solutions = clean_and_verify_safe(mat, list(mat.free_symbols))
        # hps = l.union({Hyperplane(hp, symbols) for hp in solutions})
        # print(f'my hps: {l.union(l3)}')

        # l = set()
        solutions = [tuple(*sol.items()) for sol in sp.solve(sp.simplify(mat.det()))]
        hps = l.union({Hyperplane(lhs - rhs, symbols) for lhs, rhs in solutions})
        # hps = l.union(hps)
        # print(f'real hps: {hps}')

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
        for mat in tqdm(self.cmf.matrices.values()):
            filtered = self.extract_matrix_hps(mat, self.shift, self.symbols)
            filtered_hps.update(set(filtered))
        # Logger(f'number of found hyperplanes: {len(filtered_hps)}')
        return filtered_hps

    # @staticmethod
    # def _shard_solver(enc, cmf, const, hps, shift):
    #     A, b, syms = Shard.generate_matrices(list(hps), enc)
    #     if (shard := Shard(cmf, const, A, b, shift, syms)).is_valid:
    #         return shard
    #     return None

    def extract_searchables(self) -> List[Shard]:
        """
        Extracts the shards from the CMF
        :return: The set of shards matching the CMF
        """
        hps = self.extract_cmf_hps()
        shards = []

        # shards_encodings = itertools.product((-1, 1), repeat=len(hps))
        # A, b, syms = Shard.generate_matrices(list(hps), next(shards_encodings))
        # shards_encodings = Shard.find_regions_in_box(A, b)
        # converted = []
        # for enc in shards_encodings:
        #     converted.append(tuple(1 if e else -1 for e in enc))
        # shards_encodings = converted
        # skipped = 0
        ws = []
        bs = []
        for hp in hps:
            w, b = hp.vectors
            ws.append(w)
            bs.append(b)

        # for hp in hps:
        #     print(hp)
        Logger(
            f'Found {len(hps)} hyperplanes',
            level=Logger.Levels.info
        ).log(msg_prefix='\n')

        symbols = list(hps)[0].symbols
        points = [tuple(coord + shift for coord, shift in zip(p, self.shift.values())) for p in list(itertools.product(tuple(list(range(-2, 3))), repeat=len(symbols)))]
        shard_encodings = set()
        for p in tqdm(points, desc='Checking shard encodings ...', **sys_config.TQDM_CONFIG):
            enc = []
            for hp in hps:
                res = hp.expr.subs({sym: coord for sym, coord in zip(symbols, p)})
                if res == 0:
                    break
                enc.append(1 if res > 0 else -1)
            if len(enc) == len(hps):
                shard_encodings.add(tuple(enc))
        # with Logger.simple_timer(f'extracting shard encodings'):
        #     shards_encodings = HyperplaneArrangement(np.vstack(ws), np.array(bs)).find_all_cones()
        # results = self.pool.map(partial(self._shard_solver, cmf=self.cmf, const=self.const, hps=hps, shift=self.shift), shards_encodings)
        # for res in results:
        #     if res is None:
        #         skipped += 1
        #     else:
        #         shards.append(res)
        
        for enc in tqdm(shard_encodings, desc='Creating shard objects', **sys_config.TQDM_CONFIG):
            A, b, syms = Shard.generate_matrices(list(hps), enc)
            shards.append(Shard(self.cmf, self.const, A, b, self.shift, syms))
            # if (shard := Shard(self.cmf, self.const, A, b, self.shift, syms)).is_valid:
            #     shards.append(shard)
            # else:
            #     skipped += 1
        # Logger(
        #     f'skipped {skipped} shards',
        #     level=Logger.Levels.warning
        # ).log(msg_prefix='\n')
        # for shard in shards:
        #     print(f'start point: {shard.start_coord}')
        Logger(
            f'Found {len(shard_encodings)} shards',
            level=Logger.Levels.info
        ).log(msg_prefix='\n')
        return shards


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
