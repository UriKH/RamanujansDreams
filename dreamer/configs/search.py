from dataclasses import dataclass
from typing import Callable
from .configurable import Configurable
from typing import List, Tuple
import math


def traj_from_dim(dim: int) -> int:
    return 10 ** dim


def depth_from_len(traj_len, dim) -> int:
    return min(round(1500 / max(traj_len / math.sqrt(dim), 1)), 1500)


@dataclass
class SearchConfig(Configurable):
    PARALLEL_SEARCH: bool = True
    SEARCH_VECTOR_CHUNK: int = 4                # number of search vectors per chunk for parallel search
    NUM_TRAJECTORIES_FROM_DIM: Callable = traj_from_dim
    DEPTH_FROM_TRAJECTORY_LEN: Callable = depth_from_len
    DEPTH_CONVERGENCE_THRESHOLD: Tuple[float] = (0.9, 0.95, 1.0)
    DEFAULT_USES_INV_T: bool = True

    # ============================== Delta calculation and validation settings ==============================
    LIMIT_DIFF_ERROR_BOUND: float = 1e-10           # convergence limit difference thresholds
    MIN_ESTIMATE_DENOMINATOR: int = 1e6             # estimated = a / b (if b is too small, probably didn't converge)
    CACHE_ACCEPTANCE_THRESHOLD: float = 1e-12       # p,q vector cache acceptance threshold
    IDENTIFY_CHECK_THRESHOLD: float = 1e-10


search_config: SearchConfig = SearchConfig()
