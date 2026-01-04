"""
Extractor is responsible for shard creation
"""
from dreamer.configs import (
    sys_config,
    extraction_config
)
from concurrent.futures import ProcessPoolExecutor
from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.extraction.shard import Shard
from dreamer.utils.schemes.extraction_scheme import ExtractionScheme, ExtractionModScheme
from dreamer.utils.logger import Logger
from dreamer.utils.constants.constant import Constant
from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.storage.exporter import Exporter
from dreamer.utils.storage.formats import Formats
from ramanujantools.cmf.d_finite import theta
from dreamer.utils.types import *
from dreamer.utils.ui.tqdm_config import SmartTQDM

import itertools
import os.path
import sympy as sp
from collections import defaultdict


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
                    extractor = ShardExtractor(const, cmf_shift.cmf, cmf_shift.shift)
                    shards = extractor.extract_searchables(i + 1)
                    all_shards[const] += shards
                    export_stream(shards)
        return all_shards


class ShardExtractor(ExtractionScheme):
    """
    Shard extractor is a representation of a shard finding method.
    """

    def __init__(self, const: Constant, cmf: CMF, shift: Position):
        """
        Extracts the shards of a CMF
        :param const: Constant searched in this CMF
        :param cmf: CMF to extract shards from
        :param shift: The start point shift
        """
        super().__init__(const, cmf, shift)
        self.pool = ProcessPoolExecutor() if extraction_config.PARALLELIZE else None

    @property
    def symbols(self) -> List[sp.Symbol]:
        """
        :return: The CMF's symbols
        """
        return list(self.cmf.matrices.keys())

    def extract_cmf_hps(self) -> Set[Hyperplane]:
        """
        Compute the hyperplanes of the CMF - zeros of the characteristic polynomial of each matrix and the poles of each
         matrix entry.
        :return: A set of all filtered hyperplanes (i.e., hyperplanes with respect to the shift)
        """
        hps = set()
        symbols = list(self.cmf.matrices.keys())
        for s in symbols:
            zeros = sp.solve(determinant_from_char_poly(self.cmf.p, self.cmf.q, self.cmf.z, s))
            zeros = [Hyperplane(lhs - rhs, symbols) for solution in zeros for lhs, rhs in solution.items()]
            hps.update(set(zeros))

            poles = set()
            for v in self.cmf.matrices[s].iter_values():
                if (den := v.as_numer_denom()[1]) == 1:
                    continue

                solutions = {(sym, sol) for sym in den.free_symbols for sol in sp.solve(sp.simplify(den), sym)}
                for lhs, rhs in solutions:
                    poles.add(Hyperplane(lhs - rhs, symbols))
            hps.update(poles)

        # compute the relevant hyperplanes with respect to the shift
        filtered_hps = set()
        for hp in hps:
            if hp.apply_shift(self.shift).is_in_integer_shift():
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
                Shard(self.cmf, self.const, None, None, self.shift, self.symbols)
            ]
        symbols = list(hps)[0].symbols
        points = [
            tuple(coord + shift for coord, shift in zip(p, self.shift.values()))
            for p in list(itertools.product(tuple(list(range(-2, 3))), repeat=len(symbols)))
        ]

        # validate shards using the sampled points
        shard_encodings = dict()
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
            shards.append(Shard(self.cmf, self.const, A, b, self.shift, syms, shard_encodings[enc]))
        return shards


# TODO: This method is temporary only, it should be fixed in Ramanujan tools and not here
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
