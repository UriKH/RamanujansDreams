from numba import njit, types
from numba.typed import Dict
import numpy as np
import multiprocessing as mp
import itertools
from dreamer.utils.logger import Logger


_worker_cache = {}


def __generate_numba_worker(M):
    """
    Creates the appropriate numba worker for initial point generation
    :param M: Length of the shard signature
    :return: Unique mapping between signature and initial point
    """
    num_chunks = (M + 63) // 64  # each chunk may use up to 64 bits for signature
    chunk_size = 512  # make sure size fits in L1 cache for fast computing
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

    BLOCK_SIZE = {chunk_size}
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
            for d in range(D - 1, K - 1, -1):   # Advance the state to smallest step
                state[d] += 1
                if state[d] < S: break 
                else: state[d] = 0 

        for i in range(current_batch_size):
            is_on_hyperplane = False
            sig_chunks = np.zeros({num_chunks}, dtype=np.int64)

            # Compute and update chunk signature
            for j in range(M_val):
                # substitute in the linear equation
                val = b[j]
                for d in range(D):
                    val += A[j, d] * block[i, d]

                if val == 0:
                    is_on_hyperplane = True
                    break

                if val > 0:
                    # Update chunk signature - add a 1 bit
                    chunk_idx = j // 64
                    bit_idx = np.int64(j % 64)
                    sig_chunks[chunk_idx] |= (np.int64(1) << bit_idx)

            if is_on_hyperplane:
                continue

            sig_tuple = {tuple_str}
            if sig_tuple not in unique_mapping:
                # Add chunk signature if not initialized
                unique_mapping[sig_tuple] = block[i, :].copy()
    return unique_mapping
    """
    local_env = {
        'np': np,
        'Dict': Dict,
        'tuple_type': types.UniTuple(types.int64, num_chunks),
        'int_array_type': types.int64[:]  # Use strictly int64 for the saved points
    }
    exec(code, local_env)
    return njit(local_env['dynamic_compute_block'])


def decode_signatures(unique_tuples, M):
    """
    Decode numerical signatures into +1/-1
    :param unique_tuples: A list of (encoding, point)
    :param M: Number of hyperplanes
    :return: The decoded signatures
    """
    N = len(unique_tuples)
    if N == 0:
        return np.empty((0, M), dtype=np.int8)

    chunks_array = np.array(list(unique_tuples), dtype=np.int64)
    if chunks_array.ndim == 1:
        chunks_array = chunks_array.reshape(-1, 1)

    # Convert to bits matrix
    bits = np.zeros((N, M), dtype=np.int8)
    for j in range(M):
        chunk_idx = j // 64
        bit_idx = np.int64(j % 64)
        bit_val = (chunks_array[:, chunk_idx] >> bit_idx) & np.int64(1)
        bits[:, j] = bit_val

    # Convert matrix to +1/-1 matrix
    return (bits * 2) - 1


def __worker_wrapper(fixed_prefix, D, S, A, b):
    """
    Compiles and runs the relevant numba generator
    :param fixed_prefix: Dimension reduction prefix
    :param D: Hypercube total dimensions
    :param S: Hypercube side length
    :param A: Linear equations expression matrix
    :param b: Linear equations free variables vector
    :return: A mapping from shard signature to an initial point
    """
    global _worker_cache

    M = A.shape[0]
    if M not in _worker_cache:
        _worker_cache[M] = __generate_numba_worker(M)

    compiled_func = _worker_cache[M]
    numba_dict = compiled_func(fixed_prefix, D, S, A, b)
    return {key_tuple: np.array(point_array) for key_tuple, point_array in numba_dict.items()}


def compute_mapping(D, S, A, b, prefix_dims=2):
    """
    Computes a mapping from shard signature to an initial point.
    :param D: Dimension of the hypercube
    :param S: Side length of the hypercube
    :param A: Linear equations expression matrix
    :param b: Linear equations free variables vector
    :param prefix_dims: Manually compute prefix_dims out of the D of the points
    :return: A mapping from shard signature to an initial point
    """
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
        results = pool.starmap(__worker_wrapper, tasks)

        # Merge dictionaries from all workers
        for local_mapping in results:
            for sig, point in local_mapping.items():
                if sig not in global_mapping:
                    global_mapping[sig] = point
    return global_mapping
