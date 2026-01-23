import time

import numpy as np

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
from .pfq_symetry_helper import solve_grouping, find

from concurrent.futures import ProcessPoolExecutor
import itertools
import os.path
import sympy as sp
from collections import defaultdict
from ramanujantools.cmf.d_finite import theta
from numba import njit, prange


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

        Logger(
            f'In CMF no. {call_number}: Extracting shards '
            f'(this could take a bit |>_<|, you can run with PROFILE=True to check times)',
            level=Logger.Levels.info
        ).log()
        # compute hyperplanes and prepare sample point
        with Logger.simple_timer(f'Extract CMF hyperplanes'):
            hps = self.extract_cmf_hps()
        if not hps:
            return [
                Shard(self.cmf, self.const, None, None, self.shift, self.symbols)
            ]
        symbols = list(hps)[0].symbols

        def generate_grid_numpy(values, dim):
            """
            Generates a Cartesian product grid of 'values' repeated 'dim' times.
            Equivalent to: np.array(list(itertools.product(values, repeat=dim)))
            """
            values = np.asarray(values)
            n_values = len(values)
            n_points = n_values ** dim

            out = np.empty((n_points, dim), dtype=np.float64)
            for i in range(dim):
                # Calculate repeat frequencies
                repeat_freq = n_values ** (dim - 1 - i)
                tile_freq = n_values ** i

                pattern = np.repeat(values, repeat_freq)
                out[:, i] = np.tile(pattern, tile_freq)
            return out

        # prepare sample points
        with Logger.simple_timer(f'Generate points in grid'):
            points_np = generate_grid_numpy(np.arange(-2, 3), len(symbols))
            shift_ordered = [self.shift[sym] for sym in symbols]
            shift_np = np.array(shift_ordered, dtype=np.float64)
            points_np += shift_np

        with Logger.simple_timer(f'Encode points'):
            # prepare compute
            linear_terms = []
            bias_terms = []
            for hp in hps:
                lin, b = hp.vectors
                linear_terms.append(lin.astype(np.float64))
                bias_terms.append(float(b))

            # fast compute point encodings
            hps_np = np.vstack(linear_terms)
            bias_np = np.array(bias_terms, dtype=np.float64)
            encodings, mask = compute_encodings(points_np, hps_np, bias_np)
            valid_points = points_np[mask]
            valid_encs = encodings[mask]

            # clean memory as soon as possible
            del points_np
            del encodings
            del mask

            # group points by encodings
            sort_order = np.lexsort(valid_encs.T)
            sorted_points = valid_points[sort_order]
            sorted_encs = valid_encs[sort_order]
            diff_mask = np.any(sorted_encs[1:] != sorted_encs[:-1], axis=1)
            split_indices = np.flatnonzero(diff_mask) + 1
            grouped_points_arrays = np.split(sorted_points, split_indices)
            group_starts = np.concatenate(([0], split_indices))
            unique_keys = sorted_encs[group_starts]

            del sorted_points
            del sorted_encs

            shard_point_map_stacked = {
                tuple(key): group_pts
                for key, group_pts in zip(unique_keys, grouped_points_arrays)
            }

        # preform pFq symetry based shard reduction
        if isinstance(self.cmf, pFq):
            # extract shard encodings from UF
            with Logger.simple_timer(f'Reduce shards using symetry'):
                solved = solve_grouping(list(shard_point_map_stacked.values()), self.cmf.p, self.cmf.q)
                index_shard_map = list(shard_point_map_stacked.keys())
                shard_encs = set()
                for i in range(len(index_shard_map)):
                    shard_id = find(solved, i)
                    shard_encs.add(index_shard_map[shard_id])

            # create shard objects
            with Logger.simple_timer(f'Create shards'):
                shards = []
                for enc in shard_encs:
                    A, b, syms = Shard.generate_matrices(list(hps), enc)
                    shards.append(
                        Shard(
                            self.cmf, self.const, A, b, self.shift, syms,
                            Position(zip(symbols, [sp.nsimplify(x, rational=True) for x in shard_point_map_stacked[enc][0]]))
                        )
                    )
            Logger(
                f'In CMF no. {call_number}: found {len(hps)} hyperplanes and {len(shard_point_map_stacked)} shards.'
                f' After reduction {len(shards)}',
                level=Logger.Levels.info
            ).log()
            return shards

        # create shard objects
        with Logger.simple_timer(f'Create shards'):
            shards = []
            for enc in SmartTQDM(shard_point_map_stacked.keys(), desc='Creating shard objects', **sys_config.TQDM_CONFIG):
                A, b, syms = Shard.generate_matrices(list(hps), enc)
                shards.append(
                    Shard(self.cmf, self.const, A, b, self.shift, syms, Position(zip(symbols, shard_point_map_stacked[enc])))
                )
        Logger(
            f'In CMF no. {call_number}: found {len(hps)} hyperplanes and {len(shards)} shards ',
            level=Logger.Levels.info
        ).log()
        return shards


# TODO: This method is temporary only, it should be fixed in Ramanujan tools and not here
def determinant_from_char_poly(p, q, z, axis: sp.Symbol):
    # substitute in differential equation & extract free coeff of normalized characteristic poly
    # if y axis then increment the parameter
    is_y_shift = True if axis.name.startswith("y") else False
    coeff = axis - 1 if axis.name.startswith("y") else axis
    S = sp.symbols("S")  # note that for a y shift, we're calculating the char poly for S^{-1}
    theta_subs = coeff * S - coeff
    char_poly_for_S = sp.monic(pFq.differential_equation(p, q, z).subs({theta: theta_subs}),
                               S)  # how does this handle z eval?
    free_coeff = char_poly_for_S.coeff_monomial(1)  # can also just subs

    matrix_dim = char_poly_for_S.degree()
    if is_y_shift:
        return sp.factor((((-1) ** matrix_dim) / free_coeff).subs({axis: axis + 1}))
    else:
        return sp.factor((-1) ** matrix_dim * free_coeff)


@njit(parallel=True)
def compute_encodings(points, hps_matrix, bias_vector):
    """
    Computes the sign signature for every point against all hyperplanes.
    :param points: The array of points (N, D) float64
    :param hps_matrix: Hyperplane linear term part matrix (L, D)  float64
    :param bias_vector: The hyperplane free term part (L,) float64
    :returns:
        Encodings: The encodings for each of the points as a (N, K) array of int8 (1 or -1).
        Validity: A validity flag per encoding as a (N,) boolean mask (False if the point is on a hyperplane).
    """
    n_points = points.shape[0]
    n_hps = hps_matrix.shape[0]

    encodings = np.empty((n_points, n_hps), dtype=np.int8)
    validity = np.empty(n_points, dtype=np.bool_)

    # Parallel loop over all points
    for i in prange(n_points):
        is_valid = True

        for k in range(n_hps):
            dot = 0.0
            for d in range(hps_matrix.shape[1]):
                dot += hps_matrix[k, d] * points[i, d]
            val = dot + bias_vector[k]

            if -1e-9 < val < 1e-9:
                is_valid = False
                break
            encodings[i, k] = 1 if val > 0 else -1
        validity[i] = is_valid
    return encodings, validity


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
