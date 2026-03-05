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
from numba import njit, types
from numba.typed import Dict
import numpy as np
import multiprocessing as mp
import itertools
import math


import numpy as np
from numba import njit, types
from numba.typed import Dict


def generate_numba_worker(M):
    num_chunks = (M + 63) // 64
    tuple_elements = ", ".join([f"sig_chunks[{i}]" for i in range(num_chunks)])
    tuple_str = f"({tuple_elements},)" if num_chunks == 1 else f"({tuple_elements})"

    code = f"""
def dynamic_compute_block(fixed_prefix, D, S, A, b):
    M_val = A.shape[0]
    K = len(fixed_prefix)
    rem_D = D - K  

    state = np.zeros(D, dtype=np.int32)
    for i in range(K):
        state[i] = fixed_prefix[i]

    offset = S // 2

    unique_mapping = Dict.empty(
        key_type=tuple_type,
        value_type=int_array_type  # Ensure we save int arrays
    )

    BLOCK_SIZE = 1024
    block = np.zeros((BLOCK_SIZE, D), dtype=np.int64) # Pure integer block!
    total_points = np.int64(S) ** np.int64(rem_D) 
    points_generated = np.int64(0)

    while points_generated < total_points:
        current_batch_size = 0

        while current_batch_size < BLOCK_SIZE and points_generated < total_points:
            for j in range(D):
                block[current_batch_size, j] = state[j] - offset
            current_batch_size += 1
            points_generated += 1
            for d in range(D - 1, K - 1, -1):
                state[d] += 1
                if state[d] < S: break 
                else: state[d] = 0 

        for i in range(current_batch_size):
            is_on_hyperplane = False
            sig_chunks = np.zeros({num_chunks}, dtype=np.int64)

            for j in range(M_val):
                val = b[j]
                for d in range(D):
                    val += A[j, d] * block[i, d]

                # Pure integer math check - mathematically flawless
                if val == 0:
                    is_on_hyperplane = True
                    break

                if val > 0:
                    chunk_idx = j // 64
                    bit_idx = np.int64(j % 64)
                    sig_chunks[chunk_idx] |= (np.int64(1) << bit_idx)

            if is_on_hyperplane:
                continue

            sig_tuple = {tuple_str}

            if sig_tuple not in unique_mapping:
                unique_mapping[sig_tuple] = block[i, :].copy()

    return unique_mapping
"""
    local_env = {
        'np': np,
        'Dict': Dict,
        'tuple_type': types.UniTuple(types.int64, num_chunks),
        'int_array_type': types.int64[:] # Use strictly int64 for the saved points
    }
    exec(code, local_env)
    return njit(local_env['dynamic_compute_block'])


def decode_signatures(unique_tuples, M):
    N = len(unique_tuples)
    if N == 0:
        return np.empty((0, M), dtype=np.int8)

    chunks_array = np.array(list(unique_tuples), dtype=np.int64)
    if chunks_array.ndim == 1:
        chunks_array = chunks_array.reshape(-1, 1)

    bits = np.zeros((N, M), dtype=np.int8)
    for j in range(M):
        chunk_idx = j // 64
        bit_idx = np.int64(j % 64)
        bit_val = (chunks_array[:, chunk_idx] >> bit_idx) & np.int64(1)
        bits[:, j] = bit_val

    return (bits * 2) - 1

_worker_cache = {}


def worker_wrapper(fixed_prefix, D, S, A, b):
    M = A.shape[0]

    # If this worker core hasn't compiled the function for this M yet, do it now
    if M not in _worker_cache:
        _worker_cache[M] = generate_numba_worker(M)

    compiled_func = _worker_cache[M]
    numba_dict = compiled_func(fixed_prefix, D, S, A, b)

    # Safely convert to a Python dictionary for transport back to the master process
    standard_dict = {}
    for key_tuple, point_array in numba_dict.items():
        standard_dict[key_tuple] = np.array(point_array)

    return standard_dict


def parallel_hypercube_run_with_points(D, S, A, b, prefix_dims=2):
    prefix_dims = min(prefix_dims, D)
    coords = range(S)
    prefixes = list(itertools.product(coords, repeat=prefix_dims))

    tasks = []
    for prefix in prefixes:
        prefix_arr = np.array(prefix, dtype=np.int32)
        tasks.append((prefix_arr, D, S, A, b))

    # Global dictionary to hold the final results
    global_mapping = {}

    num_cores = mp.cpu_count()
    Logger(f"Launching {len(tasks)} jobs across {num_cores} cores...").log()

    with mp.Pool(num_cores) as pool:
        results = pool.starmap(worker_wrapper, tasks)

        # Merge dictionaries from all workers
        for local_mapping in results:
            for sig, point in local_mapping.items():
                if sig not in global_mapping:
                    global_mapping[sig] = point

    return global_mapping


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
        self.pool = create_pool() if extraction_config.PARALLELIZE else None

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
            zeros = sp.solve(pFq.determinant(self.cmf_data.cmf.p, self.cmf_data.cmf.q, self.cmf_data.cmf.z, s))
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
        generated = []
        shard_encodings = dict()
        selected = [] if self.cmf_data.selected_points is None else self.cmf_data.selected_points
        if self.cmf_data.only_selected:
            if self.cmf_data.selected_points is None:
                raise ValueError('No start points were provided for extraction.')

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
        else:
            hps_list = list(hps)
            shifted_hps = [hp.apply_shift(self.cmf_data.shift) for hp in hps_list]
            A = np.array([hp.vectors[0] for hp in shifted_hps], dtype=np.int64)
            b = np.array([hp.vectors[1] for hp in shifted_hps], dtype=np.int64)
            S = config.extraction.BASE_EDGE_LENGTH * 2 + 1
            prefix_dims = max(min(int(round(math.log(os.cpu_count(), S))), os.cpu_count() - 1), 1)
            final_results = parallel_hypercube_run_with_points(self.cmf_data.cmf.dim(), S, A, b, prefix_dims)
            unique_sigs = list(final_results.keys())
            decoded_vectors = decode_signatures(unique_sigs, len(hps))
            for i, sig in enumerate(unique_sigs):
                sign_vector = decoded_vectors[i]
                if 0 in sign_vector:
                    continue
                actual_point = final_results[sig]
                shard_encodings[tuple(sign_vector)] = Position(
                    {sym: int(v) + self.cmf_data.shift[sym] for sym, v in zip(symbols, actual_point)}
                )

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
