from dreamer.configs import (
    sys_config,
    extraction_config
)
from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.extraction.shard import Shard
from dreamer.utils.schemes.extraction_scheme import ExtractionScheme, ExtractionModScheme
from dreamer.utils.logger import Logger
from dreamer.utils.constants.constant import Constant
from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.storage.exporter import Exporter
from dreamer.utils.storage.formats import Formats
from dreamer.utils.types import *
from dreamer.utils.ui.tqdm_config import SmartTQDM
from dreamer.configs import config
from dreamer.utils.mp_manager import create_pool

import os.path
import sympy as sp
from collections import defaultdict
from numba.typed import Dict
import math
import numpy as np
from .utils import initial_points as init_points
from functools import partial
from ramanujantools.cmf import pFq as rt_pFq


class ShardExtractorMod(ExtractionModScheme):
    """
    Module for shard extraction
    """

    def __init__(self, cmf_data: Dict[Constant, List[ShiftCMF]]):
        """
        Creates a shard extraction module
        :param cmf_data: A mapping from constants to a list of CMFs
        """
        super().__init__(
            cmf_data,
            name=self.__class__.__name__,
            desc='Shard extractor module',
            version='0.0.1'
        )

    def execute(self) -> Dict[Constant, List[Searchable]]:
        """
        Extract shards from CMFs
        :return: A mapping from constants to a list of shards
        """
        all_shards = defaultdict(list)

        consts_itr = iter(list(self.cmf_data.keys()))
        for const, cmf_list in SmartTQDM(
                self.cmf_data.items(), desc=f'Extracting shards for "{next(consts_itr).name}"',
                **sys_config.TQDM_CONFIG
        ):
            with Exporter.export_stream(
                    os.path.join(extraction_config.PATH_TO_SEARCHABLES, const.name),
                    exists_ok=True, clean_exists=True, fmt=Formats.PICKLE
            ) as export_stream:
                for i, cmf_shift in enumerate(SmartTQDM(
                        cmf_list, desc=f'Computing shards',
                        **sys_config.TQDM_CONFIG)):
                    extractor = ShardExtractor(
                        const, cmf_shift
                    )
                    shards = extractor.extract_searchables(call_number=i + 1)
                    all_shards[const] += shards
                    export_stream(shards)
        return all_shards


class ShardExtractor(ExtractionScheme):
    """
    Shard extractor is a representation of a shard finding method.
    """

    def __init__(self, const: Constant, cmf_data: ShiftCMF):
        """
        Extracts the shards of a CMF
        :param const: Constant searched in this CMF
        :param cmf_data: CMF to extract shards from, more data for extraction and later usage
        """
        super().__init__(const, cmf_data)
        # self.pool = create_pool() if extraction_config.PARALLELIZE else None

    @property
    def symbols(self) -> List[sp.Symbol]:
        """
        :return: The CMF's symbols
        """
        return list(self.cmf_data.cmf.matrices.keys())

    def extract_cmf_hps(self) -> Set[Hyperplane]:
        """
        Compute the hyperplanes of the CMF - zeros of the characteristic polynomial of each matrix and the poles of each
         matrix entry.
        :return: A set of all filtered hyperplanes (i.e., hyperplanes with respect to the shift)
        """
        hps = set()
        symbols = list(self.cmf_data.cmf.matrices.keys())
        for s in symbols:
            if isinstance(self.cmf_data.cmf, rt_pFq):
                det = rt_pFq.determinant(self.cmf_data.cmf.p, self.cmf_data.cmf.q, self.cmf_data.cmf.z, s)
            else:
                det = self.cmf_data.cmf.matrices[s].det()
            zeros = sp.solve(det)
            zeros = [Hyperplane(lhs - rhs, symbols) for solution in zeros for lhs, rhs in solution.items()]
            hps.update(set(zeros))

            poles = set()
            for v in self.cmf_data.cmf.matrices[s].iter_values():
                if (den := v.as_numer_denom()[1]) == 1:
                    continue

                solutions = {(sym, sol) for sym in den.free_symbols for sol in sp.solve(sp.simplify(den), sym)}
                for lhs, rhs in solutions:
                    poles.add(Hyperplane(lhs - rhs, symbols))
            hps.update(poles)

        # compute the relevant hyperplanes with respect to the shift
        filtered_hps = set()
        for hp in hps:
            if hp.apply_shift(self.cmf_data.shift).is_in_integer_shift():
                filtered_hps.add(hp)
        return filtered_hps

    def extract_searchables(self, call_number=None) -> List[Shard]:
        """
        Extracts the shards from the CMF
        :return: The list of shards matching the CMF
        """
        # compute hyperplanes and prepare sample point
        hps = self.extract_cmf_hps()

        if not hps:
            return [
                Shard(self.cmf_data.cmf, self.const, None, None, self.cmf_data.shift, self.symbols,
                      use_inv_t=self.cmf_data.use_inv_t)
            ]

        symbols = list(hps)[0].symbols
        shard_encodings = dict()
        selected = [] if self.cmf_data.selected_points is None else self.cmf_data.selected_points
        if self.cmf_data.only_selected:
            if self.cmf_data.selected_points is None:
                raise ValueError('No start points were provided for extraction.')
        else:
            hps_list = list(hps)
            shifted_hps = [hp.apply_shift(self.cmf_data.shift) for hp in hps_list]
            A = np.array([hp.vectors[0] for hp in shifted_hps], dtype=np.int64)
            b = np.array([hp.vectors[1] for hp in shifted_hps], dtype=np.int64)
            S = config.extraction.INIT_POINT_MAX_COORD * 2 + 1
            prefix_dims = max(min(int(round(math.log(os.cpu_count(), S))), os.cpu_count() - 1), 1)

            symmetries_func = None
            if issubclass(self.cmf_data.cmf.__class__, rt_pFq) and config.extraction.IGNORE_DUPLICATE_SEARCHABLES:
                symmetries_func = partial(init_points.filter_symmetrical_cones,
                                          p=self.cmf_data.cmf.p,
                                          q=self.cmf_data.cmf.q,
                                          shift=list(self.cmf_data.shift.values()))
            final_results = init_points.compute_mapping(
                self.cmf_data.cmf.dim(), S, A, b, prefix_dims, symmetries_func
            )
            unique_sigs = list(final_results.keys())
            decoded_vectors = init_points.decode_signatures(unique_sigs, len(hps))
            for i, sig in enumerate(unique_sigs):
                sign_vector = decoded_vectors[i]
                if 0 in sign_vector:
                    continue
                actual_point = final_results[sig]
                shard_encodings[tuple(sign_vector)] = Position(
                    {sym: int(v) + self.cmf_data.shift[sym] for sym, v in zip(symbols, actual_point)}
                )

        if self.cmf_data.selected_points:
            points = [
                tuple(coord + shift for coord, shift in zip(p, self.cmf_data.shift.values()))
                for p in selected
            ]

            # validate shards using the sampled points
            for p in SmartTQDM(points, desc='Computing shard encodings', **sys_config.TQDM_CONFIG):
                enc = []
                point_dict = {sym: coord for sym, coord in zip(symbols, p)}
                for hp in hps:
                    res = hp.expr.subs(point_dict)
                    if res == 0:
                        break
                    enc.append(1 if res > 0 else -1)

                if len(enc) == len(hps):
                    shard_encodings[tuple(enc)] = Position(point_dict)

        Logger(
            f'In CMF no. {call_number}: found {len(hps)} hyperplanes and {len(shard_encodings)} shards ',
            level=Logger.Levels.info
        ).log()

        # create shard objects
        shards = []
        for enc in SmartTQDM(shard_encodings.keys(), desc='Creating shard objects', **sys_config.TQDM_CONFIG):
            A, b, syms = Shard.generate_matrices(list(hps), enc)
            shards.append(Shard(self.cmf_data.cmf, self.const, A, b, self.cmf_data.shift, syms, shard_encodings[enc], self.cmf_data.use_inv_t))
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
