"""
Extractor is responsible for shard creation
"""
import itertools
from rt_search.utils.types import *
import sympy as sp

from rt_search.configs import analysis_config
from concurrent.futures import ProcessPoolExecutor
from rt_search.analysis_stage.shards.hyperplanes import Hyperplane
from rt_search.analysis_stage.shards.shard import Shard
from rt_search.utils.logger import Logger


class ShardExtractor:
    def __init__(self, const_name: str, cmf: CMF, shift: Position):
        self.const_name = const_name
        self.cmf: CMF = cmf
        self.shift: Position = shift
        self.pool = ProcessPoolExecutor() if analysis_config.PARALLEL_SHARD_VALIDATION else None

    @property
    def symbols(self) -> List[sp.Symbol]:
        return list(self.cmf.matrices.keys())

    @staticmethod
    def extract_matrix_hps(mat, shift: Position, symbols: List[sp.Symbol]) -> List[sp.Expr]:
        """
        Extracts HPs from the matrix with respect to shift (ignore irrelevant HPs)
        :param mat: Matrix to extract HPs from
        :param shift: shift for HPs
        :param symbols: symbols used by the CMF
        :return: set of hyperplanes filtered and shifted
        """
        # Zero division solutions
        l = set()

        for v in mat.iter_values():
            if (den := v.as_numer_denom()[1]) == 1:
                continue
            solutions = {(sym, sol) for sym in den.free_symbols for sol in sp.solve(den, sym)}
            for lhs, rhs in solutions:
                l.add(Hyperplane(lhs - rhs, symbols))

        # Zero det solutions
        solutions = [tuple(*sol.items()) for sol in sp.solve(mat.det())]
        hps = l.union({Hyperplane(lhs - rhs, symbols) for lhs, rhs in solutions})

        hps = hps
        filtered_hps = []
        for hp in hps:
            shifted = hp.apply_shift(shift)
            if shifted.is_in_integer_shift():
                filtered_hps.append(hp)
        return filtered_hps

    def extract_cmf_hps(self):
        """
        Compute the hyperplanes of the CMF a shifted trajectory could encounter
        :return: two sets - the filtered hyperplanes and the shifted hyperplanes
        """
        filtered_hps = set()
        for mat in self.cmf.matrices.values():
            filtered = self.extract_matrix_hps(mat, self.shift, self.symbols)
            filtered_hps.update(set(filtered))
        return filtered_hps

    def extract_shards(self) -> List[Shard]:
        """
        Extracts the shards from the CMF
        :return: The set of shards matching the CMF
        """
        hps = self.extract_cmf_hps()
        shards = []

        def bit_comb_generator(n):
            i = 0
            while i < 2 ** n:
                s = bin(i)[2:].zfill(n)
                yield [v if (v := int(bit)) == 1 else -1 for bit in s.split('')]
                i += 1
            return None

        shards_encodings = itertools.product((-1, 1), repeat=len(hps)) # TODO: This might take a long time!
        # TODO: while enc := bit_comb_generator(len(self.symbols)):
        skipped = 0
        for enc in shards_encodings:
            A, b, syms = Shard.generate_matrices(list(hps), enc)
            if (shard := Shard(self.cmf, self.const_name, A, b, self.shift, syms)).is_valid:
                shards.append(shard)
            else:
                skipped += 1
            # else:
        Logger(
            f'skipped {skipped} shards in CMF {self.cmf} with shift {self.shift}',
            level=Logger.Levels.warning
        ).log(msg_prefix='\n')
        return shards


if __name__ == '__main__':
    x0, x1, y0, n = sp.symbols('x0 x1 y0 n')
    from ramanujantools.cmf import pFq
    # This is pi 2F1 CMF
    pi = pFq(2, 1, sp.Rational(1, 2))

    shift = Position({x0: sp.Rational(1, 2), x1: sp.Rational(1,2), y0: sp.Rational(1,2)})
    from pprint import pprint
    # pprint(ShardExtractor('pi', pi, shift).extract_cmf_hps())
    # ppt = ShardExtractor('pi', pi, shift).extract_shards()
    # pprint(len(ppt))
    shifted = Hyperplane(x0+1, [x0, x1, y0]).apply_shift(shift)
    print(shifted.expr)
    print(shifted.is_in_integer_shift())
