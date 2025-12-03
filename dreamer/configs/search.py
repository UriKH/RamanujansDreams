from dataclasses import dataclass

from .configurable import Configurable


@dataclass
class SearchConfig(Configurable):
    PARALLEL_SEARCH: bool = True
    SEARCH_VECTOR_CHUNK: int = 4                # number of search vectors per chunk for parallel search
    WARN_ON_EMPTY_SHARDS: bool = False          # warn user if start points could not be found in the shards
    NUM_TRAJECTORIES_FROM_DIM: callable = (lambda dim: 10 ** dim)


search_config: SearchConfig = SearchConfig()
