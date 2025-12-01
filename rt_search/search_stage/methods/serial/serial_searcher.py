from ...data_manager import *
from ...searcher_scheme import SearchMethod
from rt_search.utils.types import *
from rt_search.utils.logger import Logger
from rt_search.configs import search_config

import sympy as sp
import mpmath as mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from rt_search.analysis_stage.shards.searchable import Searchable


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
                 use_LIReC: bool = True,
                 deep_search: bool = True):     # TODO: we don't use deep_search and share_data, for latewr usage...
        """
        Creates a searcher
        :param space: The space to search in.
        """
        super().__init__(space, constant, use_LIReC, data_manager, share_data, deep_search)
        self.trajectories: Set[Position] = set()
        self.data_manager = data_manager if data_manager else DataManager(use_LIReC)
        self.const_name = space.const_name
        self.parallel = ((not self.deep_search and search_config.PARALLEL_TRAJECTORY_MATCHING)
                         or search_config.PARALLEL_SEARCH)
        self.pool = ProcessPoolExecutor() if self.parallel else None

    def generate_trajectories(self, n: int):
        self.trajectories = self.space.sample_trajectories(n, strict=False)

    def search(self,
               starts: Optional[Position | List[Position]] = None,
               partial_search_factor: float = 1,
               find_limit: bool = True,
               find_eigen_values: bool = True,
               find_gcd_slope: bool = True) -> DataManager:
        if partial_search_factor > 1 or partial_search_factor < 0:
            raise ValueError("partial_search_factor must be between 0 and 1")
        if not starts:
            starts = self.space.get_interior_point()
        if isinstance(starts, Position):
            starts = [starts]

        trajectories = self.trajectories
        if partial_search_factor < 1:
            trajectories = set(self.pick_fraction(self.trajectories, partial_search_factor))
            if len(trajectories) == 0:
                Logger(
                    'Too few trajectories, all chosen for search (consider adjusting partial_search_factor)',
                    Logger.Levels.warning
                ).log()
                trajectories = self.trajectories

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

    def get_data(self):
        """
        :return:
        """
        """
        return self.data_manager.get_data()
        """
        raise NotImplementedError

    def enrich_trajectories(self):
        raise NotImplementedError

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
