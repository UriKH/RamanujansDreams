from dataclasses import dataclass

from .configurable import Configurable


@dataclass
class SearchConfig(Configurable):
    PARALLEL_SEARCH: bool = True
    SEARCH_VECTOR_CHUNK: int = 4                # number of search vectors per chunk for parallel search
    NUM_TRAJECTORIES_FROM_DIM: callable = (lambda dim: 10 ** dim)
    DEPTH_FROM_TRAJECTORY_LEN: callable = (lambda traj_len: min(int(1000 / (traj_len / 5)), 1000))


search_config: SearchConfig = SearchConfig()
