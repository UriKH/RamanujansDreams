from dataclasses import dataclass
from typing import Callable
from .configurable import Configurable
from typing import List


@dataclass
class SearchConfig(Configurable):
    PARALLEL_SEARCH: bool = True
    SEARCH_VECTOR_CHUNK: int = 4                # number of search vectors per chunk for parallel search
    NUM_TRAJECTORIES_FROM_DIM: Callable = (lambda dim: 10 ** dim)
    DEPTH_FROM_TRAJECTORY_LEN: Callable = (lambda traj_len: min(round(1000 / (traj_len / 5)), 1000))

    # ============================== Delta calculation and validation settings ==============================
    LIMIT_DIFF_ERROR_BOUND: float = 1e-10           # convergence limit difference thresholds
    MIN_ESTIMATE_DENOMINATOR: int = 1e6             # estimated = a / b (if b is too small, probably didn't converge)
    CACHE_ACCEPTANCE_THRESHOLD: float = 1e-12       # p,q vector cache acceptance threshold
    IDENTIFY_CHECK_THRESHOLD: float = 1e-10


search_config: SearchConfig = SearchConfig()
