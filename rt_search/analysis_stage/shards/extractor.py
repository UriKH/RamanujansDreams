"""
Extractor is responsible for shard creation
"""
import itertools
from collections import defaultdict
from rt_search.utils.caching import *

from ramanujantools.position import Position
from typing import Tuple, Dict, List, Set
import sympy as sp

from rt_search.configs import (
    sys_config,
    analysis_config
)

from rt_search.utils.cmf import CMF
from concurrent.futures import ProcessPoolExecutor
from rt_search.analysis_stage.shards.hyperplanes import Hyperplane
from rt_search.analysis_stage.shards.shard import Shard


class ShardExtractor:
    def __init__(self, const_name: str, cmf: CMF, shift: Position):
        self.const_name = const_name
        self.cmf: CMF = cmf
        self.shift: Position = shift
        self.pool = ProcessPoolExecutor() if analysis_config.PARALLEL_SHARD_VALIDATION else None

    @cached_property
    def symbols(self) -> List[sp.Symbol]:
        return list(self.cmf.matrices.keys())

    def compute_groups(self) -> Dict[Tuple[sp.Symbol, ...], Set[Hyperplane]]:
        """
        Computes the groups of hyperplanes given the cmf
        :return: A mapping from a group (tuple of symbols used) to the set of hyperplanes
        """
        # Generate the general shard options
        combs = itertools.combinations((0, 1), len(self.symbols))
        key_groups = [(self.symbols[ind] for ind in comb if ind == 1) for comb in combs]

        # Extract hyperplanes per matrix
        hps = {sym: self.extract_hps(self.cmf.matrices[sym], self.shift) for sym in self.symbols}

        # Find hps per group
        shards = {}
        for key_group in key_groups:
            g_hps = set()
            for k in key_group:
                g_hps.update(hps[k])
            shards[key_group] = g_hps
        return shards

    def extract_hps(self, mat, shift: Position) -> Tuple[List[sp.Expr], List[sp.Expr]]:
        """
        Extracts HPs from the matrix with respect to shift (ignore irrelevant HPs)
        :param mat: Matrix to extract HPs from
        :param shift: shift for HPs
        :return: set of hyperplanes
        """
        # Zero division solutions
        l = set()

        for v in mat.iter_values():
            if (den := v.as_numer_denom()[1]) == 1:
                continue
            solutions = [(den, sol) for sym in den.free_symbols for sol in sp.solve(den, sym)]
            for lhs, rhs in solutions:
                l.add(Hyperplane(lhs - rhs, self.symbols))

        # Zero det solutions
        solutions = [tuple(*sol.items()) for sol in sp.solve(mat.det())]
        hps = l.union({Hyperplane(lhs - rhs, self.symbols) for lhs, rhs in solutions})

        hps = hps
        filtered_hps = []
        shifted_hps = []
        for hp in hps:
            shifted = hp.apply_shift(shift)
            if shifted.is_in_integer_shift():
                filtered_hps.append(hp)
                shifted_hps.append(hp)
        return filtered_hps, shifted_hps

    def extract(self) -> Set[Shard]:
        """
        Extracts the shards from the CMF
        :return: The set of shards matching the CMF
        """
        shard_groups = self.compute_groups()
        shards = defaultdict(list)
        for g, (hps, shifted_hps) in shard_groups.items():
            shards_encodings = itertools.combinations((-1, 1), len(g))
            for enc in shards_encodings:
                A, b, syms = Shard.generate_matrices(list(hps), enc)
                if (shard := Shard(A, b, g, self.shift, syms)).is_valid:
                    shards[g].append(shard)
        unified = set()
        for v in shards.values():
            unified.update(v)
        return unified


if __name__ == '__main__':
    x0, x1, y0, n = sp.symbols('x0 x1 y0 n')
    from ramanujantools.cmf import pFq
    # This is pi 2F1 CMF
    pi = pFq(2, 1, sp.Rational(1, 2))

    print(ShardExtractor.extract_hps(pi.matrices[x0], [x0, x1, y0]))
