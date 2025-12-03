from dreamer.utils.storage.storage_objects import *
from dreamer.utils.schemes.searcher_scheme import SearchMethod
from dreamer.utils.types import *
from dreamer.utils.logger import Logger
from dreamer.configs import search_config

import pandas as pd
import sympy as sp
import mpmath as mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from dreamer.utils.schemes.searchable import Searchable


class SerialSearcher(SearchMethod):
    """
    Serial trajectory searcher. \n
    No parallelism or smart co-boundary. \n
    """

    def __init__(self,
                 space: Searchable,
                 constant,  # sympy constant or mp.mpf
                 data_manager: DataManager = None,
                 share_data: bool = True,
                 use_LIReC: bool = True):
        """
        Creates a searcher
        :param space: The space to search in.
        """
        super().__init__(space, constant, use_LIReC, data_manager, share_data)
        self.data_manager = data_manager if data_manager else DataManager(use_LIReC)
        self.const_name = space.const_name
        self.parallel = search_config.PARALLEL_SEARCH
        self.pool = ProcessPoolExecutor() if self.parallel else None

    def search(self,
               starts: Optional[Position | List[Position]] = None,
               find_limit: bool = True,
               find_eigen_values: bool = True,
               find_gcd_slope: bool = True,
               trajectory_generator: Callable[int, int] = search_config.NUM_TRAJECTORIES_FROM_DIM
               ) -> DataManager:
        if not starts:
            starts = self.space.get_interior_point()
        if isinstance(starts, Position):
            starts = [starts]

        trajectories = self.space.sample_trajectories(
            trajectory_generator(self.space.dim),
            strict=False
        )

        pairs = [(t, start) for start in starts for t in trajectories if
                 SearchVector(start, t) not in self.data_manager]
        traj_lst = [p[0] for p in pairs]
        start_lst = [p[1] for p in pairs]

        if self.parallel:
            results = self.pool.map(
                partial(
                    self.space.compute_trajectory_data,
                    use_LIReC=self.use_LIReC,
                    find_limit=find_limit,
                    find_eigen_values=find_eigen_values,
                    find_gcd_slope=find_gcd_slope
                ),
                traj_lst, start_lst, chunksize=search_config.SEARCH_VECTOR_CHUNK)
            for res in results:
                if res:
                    res.gcd_slope = mp.mpf(res.gcd_slope) if res.gcd_slope else None
                    res.delta = mp.mpf(res.delta) if isinstance(res.delta, str) else res.delta
                    self.data_manager[res.sv] = res
        else:
            for t, start in pairs:
                sd = self.space.compute_trajectory_data(
                    t, start,
                    use_LIReC=self.use_LIReC,
                    find_limit=find_limit,
                    find_eigen_values=find_eigen_values,
                    find_gcd_slope=find_gcd_slope
                )
                self.data_manager[sd.sv] = sd
        return self.data_manager

    @staticmethod
    def sympy_to_mpmath(x):
        if x is sp.zoo:
            return mp.mpf('inf')
        elif x.is_infinite:
            if x == sp.oo:
                return mp.mpf('inf')
            elif x == -sp.oo:
                return mp.mpf('-inf')
            else:
                return mp.mpf('-inf')  # zoo or directional infinity
        else:
            return mp.mpf(str(x.evalf(500)))

    @staticmethod
    def fraction_to_vectors(frac, symbols):
        """
        Convert a tuple (num, den) of sympy expressions into coefficient vectors.

        Args:
            frac: tuple (numerator_expr, denominator_expr)
            symbols: list of sympy symbols [c1, c2, ...]

        Returns:
            (num_vec, den_vec): two lists of coefficients
                Index 0 = constant term
                Index i = coefficient of symbols[i-1]
        """
        num_expr, den_expr = frac

        def expr_to_vector(expr, symbols):
            coeffs = [expr.as_coeff_add(*symbols)[0]]  # constant term
            for s in symbols:
                coeffs.append(expr.coeff(s))
            return [sp.sympify(c) for c in coeffs]

        return expr_to_vector(num_expr, symbols), expr_to_vector(den_expr, symbols)
