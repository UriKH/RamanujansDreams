from dreamer.utils.schemes.analysis_scheme import AnalyzerScheme
from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.storage.storage_objects import DataManager
from dreamer.utils.ui.tqdm_config import SmartTQDM
from dreamer.utils.constants.constant import Constant
from dreamer.utils.types import *
from dreamer.utils.logger import Logger
from dreamer.search.methods.serial.serial_searcher import SerialSearcher
from dreamer.configs import (
    sys_config,
    analysis_config
)


class Analyzer(AnalyzerScheme):
    """
    The analyzer is an implementation of a method to perform a small search of sorts and
    prioritize the searchables according to relevance.
    """

    def __init__(self, const: Constant, shards: List[Searchable]):
        """
        :param const: Constant to analyze for
        :param shards: Searchables to analyze
        """
        self.const = const
        self.shards = shards

    def search(self) -> Dict[Searchable, DataManager]:
        """
        Preforms a small search inside the shards
        :return: The results per shard
        """
        managers = {}

        for i, shard in enumerate(SmartTQDM(self.shards, desc=f'Analyzing shards', **sys_config.TQDM_CONFIG)):
            start = shard.get_interior_point()
            Logger(f'{"=" * 10} SHARD NO. {i + 1} {"=" * 10}').log(msg_prefix='\n')
            if analysis_config.SHOW_START_POINT:
                Logger(f'Chosen shard start point: {start}', Logger.Levels.info).log()
            if analysis_config.SHOW_SEARCHABLE:
                Logger(f'Shard: \n{shard}', Logger.Levels.info).log()

            searcher = SerialSearcher(shard, self.const, use_LIReC=analysis_config.USE_LIReC)
            dm = searcher.search(
                start,
                find_limit=analysis_config.ANALYZE_LIMIT,
                find_gcd_slope=analysis_config.ANALYZE_GCD_SLOPE,
                find_eigen_values=analysis_config.ANALYZE_EIGEN_VALUES,
                trajectory_generator=analysis_config.NUM_TRAJECTORIES_FROM_DIM
            )

            identified = dm.identified_percentage
            best_delta, best_trajectory = dm.best_delta

            if analysis_config.PRINT_FOR_EVERY_SEARCHABLE:
                if best_delta is None:
                    Logger(
                        f'Identified {identified * 100:.2f}% of trajectories as containing "{self.const.name}"',
                        Logger.Levels.info
                    ).log()
                else:
                    cmf_msg = f'{str(shard.cmf)}'\
                        if type(shard.cmf) is not CMF else '<raw cmf>'
                    Logger(
                        f'Identified {identified * 100:.2f}% of trajectories as containing "{self.const.name}"'
                        f'\n    > Best delta:\t {best_delta:.4f}'
                        f'\n    > Trajectory:\t {best_trajectory}'
                        f'\n    > CMF:\t\t {cmf_msg}'
                        f'\n    > p,q vectors:\t{dm[best_trajectory].initial_values.tolist()}',
                        Logger.Levels.info
                    ).log()
            if identified > analysis_config.IDENTIFY_THRESHOLD and best_delta is not None:
                managers[shard] = dm
            else:
                if best_delta is not None:
                    Logger(
                        f'Ignoring shard - identified <= {analysis_config.IDENTIFY_THRESHOLD * 100}% '
                        f'of tested trajectories',
                        Logger.Levels.info
                    ).log()
                else:
                    Logger(f'No best delta was found', Logger.Levels.warning).log()
        return managers

    def prioritize(self, managers: Dict[Searchable, DataManager], ranks: int = 3) -> Dict[Searchable, Dict[str, int]]:
        """
        Prioritizes the searchables according to how big is their delta and the CMF dimension
        :param managers: A mapping from a searchable to its search result.
        :param ranks: Number of delta ranks (we divide the interval [-1, 0^+] into #ranks).
        :return: A mapping from searchables to their prioritization object:
            ``{'delta_rank':<rank>,'dim':<searchable-dim>}``
        """
        if ranks < 3:
            Logger('prioritization ranks must be >= 3 (resulting to default = 3), '
                   'continuing to prevent data loss', Logger.Levels.inform).log()

        def match_rank(n: int, num):
            step = 1 / n
            ranks = [-1 + k * step for k in range(n + 1)]
            l = 0
            r = len(ranks) - 1

            while l < r:
                mid = (l + r) // 2
                if num > ranks[mid]:
                    l = mid + 1
                else:
                    r = mid
            return l + 1

        ranked = {}
        for shard, dm in managers.items():
            best_delta = dm.best_delta[0]
            ranked[shard] = {
                'delta_rank': match_rank(ranks, best_delta if best_delta else -1),
                'dim': shard.dim
            }
        return ranked
