"""
Extractor is responsible for shard creation
"""
import itertools

from rt_search.utils.types import *
from rt_search.utils.geometry.plane import Plane
from rt_search.configs import (
    sys_config,
    analysis_config
)

from rt_search.analysis_stage.shards.shard import Shard
from rt_search.utils.geometry.point_generator import PointGenerator
from rt_search.utils.logger import Logger
from rt_search.utils.cmf import CMF

from itertools import product
from functools import lru_cache, partial
from scipy.optimize import linprog
import numpy as np
from numpy import linalg
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from rt_search.analysis_stage.shards.hyperplanes import Hyperplane


class ShardExtractor:
    def __init__(self, const_name: str, cmf: CMF, shift: Position):
        self.const_name = const_name
        self.cmf: CMF = cmf
        self.shift: Position = shift
        self.pool = ProcessPoolExecutor() if analysis_config.PARALLEL_SHARD_VALIDATION else None
        # self.hps, self.symbols = self.__extract_shard_hyperplanes(cmf)
        # Logger(
        #     f'\n* symbols for this CMF: {self.symbols}\n* Shifts: {self.shifts}', Logger.Levels.info
        # ).log(msg_prefix='\n')
        #
        # # This will be instantiated later on the first call to get_encoded_shards()
        # self._mono_shard = len(self.hps) == 0
        # self._encoded_shards = [tuple()] if self._mono_shard else None
        # self._feasible_points = [[0] * len(self.symbols)] if self._mono_shard else None
        # self._shards = [Shard(tuple(), self)] if self._mono_shard else None
        # self._populated = False

    def compute_groups(self) -> Dict[Tuple[sp.Symbol, ...], Set[Hyperplane]]:
        """
        Computes the groups of hyperplanes given the cmf
        :return: A mapping from a group (tuple of symbols used) to the set of hyperplanes
        """
        # Generate the general shard options
        mats = self.cmf.matrices
        syms = list(mats.keys())
        combs = itertools.combinations((0, 1), len(mats))
        key_groups = [(syms[ind] for ind in comb if ind == 1) for comb in combs]

        # Extract hyperplanes per matrix
        hps = {sym: self.extract_hps(mats[sym], syms) for sym in syms}

        # Find hps per group
        shards = {}
        for key_group in key_groups:
            g_hps = set()
            for k in key_group:
                g_hps.update(hps[k])
            shards[key_group] = g_hps
        return shards

    @staticmethod
    def extract_hps(mat, symbols: List[sp.Symbol]) -> Set[sp.Expr]:
        """
        Extracts HPs from the matrix
        :param mat: Matrix to extract HPs from
        :param symbols: symbols for HP
        :return: set of hyperplanes
        """
        # Zero division solutions
        l = set()

        for v in mat.iter_values():
            if (den := v.as_numer_denom()[1]) == 1:
                continue
            solutions = [(den, sol) for sym in den.free_symbols for sol in sp.solve(den, sym)]
            for lhs, rhs in solutions:
                l.add(Hyperplane(lhs - rhs, symbols))

        # Zero det solutions
        solutions = [tuple(*sol.items()) for sol in sp.solve(mat.det())]
        return l.union({Hyperplane(lhs - rhs, symbols) for lhs, rhs in solutions})

    def extract(self):
        pass


if __name__ == '__main__':
    x0, x1, y0, n = sp.symbols('x0 x1 y0 n')
    from ramanujantools.cmf import pFq
    # This is pi 2F1 CMF
    pi = pFq(2, 1, sp.Rational(1, 2))

    print(ShardExtractor.extract_hps(pi.matrices[x0], [x0, x1, y0]))
