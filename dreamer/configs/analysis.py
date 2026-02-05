from dataclasses import dataclass
from .configurable import Configurable
from typing import Callable


def traj_from_dim(dim: int) -> int:
    return 10 ** dim


@dataclass
class AnalysisConfig(Configurable):
    """
    Stage analysis configurations
    """
    # ============================= Parallelism and efficiency =============================
    USE_CACHING: bool = True  # use caching for lru_cache

    NUM_TRAJECTORIES_FROM_DIM: Callable = traj_from_dim     # #trajectories to analyze given searchable dims
    IDENTIFY_THRESHOLD: float = -1  # consider a shard as related to a constant: threshold > identified_trajectories(%)

    # ============================= Printing and error management =============================
    PRINT_FOR_EVERY_SEARCHABLE: bool = True         # print the result of each analysis of a searchable object
    SHOW_START_POINT: bool = True
    SHOW_SEARCHABLE: bool = False

    # ============================= Analysis features =============================
    USE_LIReC: bool = True                  # use LIReC in analysis instead of RT functions
    ANALYZE_LIMIT: bool = False             # calculate the limit
    ANALYZE_EIGEN_VALUES: bool = False      # calculate the trajectory matrix eigen values
    ANALYZE_GCD_SLOPE: bool = False         # calculate the gcd slope of the trajectory


analysis_config: AnalysisConfig = AnalysisConfig()
