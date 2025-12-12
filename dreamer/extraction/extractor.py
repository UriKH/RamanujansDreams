"""
Extractor is responsible for shard creation
"""
import itertools
import os.path
import time
from collections import defaultdict
from functools import partial

from dreamer.utils.schemes.extraction_scheme import ExtractionScheme, ExtractionModScheme
from dreamer.utils.types import *
import sympy as sp
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
        time.sleep(0.5)
        for const, cmf_list in tqdm(self.cmf_data.items(), desc=f'Extracting shards for "{next(consts_itr).name}"',
                                    **sys_config.TQDM_CONFIG):
            with Exporter.export_stream(
                    os.path.join(extraction_config.PATH_TO_SEARCHABLES, const.name),
                    exists_ok=True, clean_exists=True, fmt=Formats.PICKLE
            ) as export_stream:
                ind_itr = iter(range(len(cmf_list)))
                time.sleep(0.5)
                for cmf_shift in tqdm(cmf_list, desc=f'Computing shards for CMF no. {next(ind_itr) + 1}',
                                      **sys_config.TQDM_CONFIG):
                    extractor = ShardExtractor(const, cmf_shift.cmf, cmf_shift.shift)
                    shards = extractor.extract_searchables()
                    all_shards[const] += shards
                    export_stream(shards)
                time.sleep(0.5)
        return all_shards


class ShardExtractor(ExtractionScheme):
    def __init__(self, const: Constant, cmf: CMF, shift: Position):
        super().__init__(const, cmf, shift)
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

    @staticmethod
    def _shard_solver(enc, cmf, const, hps, shift):
        A, b, syms = Shard.generate_matrices(list(hps), enc)
        if (shard := Shard(cmf, const, A, b, shift, syms)).is_valid:
            return shard
        return None

    def extract_searchables(self) -> List[Shard]:
        """
        Extracts the shards from the CMF
        :return: The set of shards matching the CMF
        """
        hps = self.extract_cmf_hps()
        shards = []

        shards_encodings = itertools.product((-1, 1), repeat=len(hps))
        A, b, syms = Shard.generate_matrices(list(hps), next(shards_encodings))
        shards_encodings = Shard.find_regions_in_box(A, b)
        converted = []
        for enc in shards_encodings:
            converted.append(tuple(1 if e else -1 for e in enc))
        shards_encodings = converted
        skipped = 0
        # shards_encodings =
        results = self.pool.map(partial(self._shard_solver, cmf=self.cmf, const=self.const, hps=hps, shift=self.shift), shards_encodings)
        for res in results:
            if res is None:
                skipped += 1
            else:
                shards.append(res)
        # for enc in shards_encodings:
        #     A, b, syms = Shard.generate_matrices(list(hps), enc)
        #     if (shard := Shard(self.cmf, self.const, A, b, self.shift, syms)).is_valid:
        #         shards.append(shard)
        #     else:
        #         skipped += 1
        Logger(
            f'skipped {skipped} shards',
            level=Logger.Levels.warning
        ).log(msg_prefix='\n')
        return shards


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
