from dreamer.utils.schemes.analysis_scheme import AnalyzerScheme
from dreamer.utils.schemes.searchable import Searchable
from dreamer.extraction.extractor import ShardExtractor
from dreamer.utils.storage.storage_objects import DataManager
from dreamer.search.methods.serial.serial_searcher import SerialSearcher
from dreamer.utils.types import *
from dreamer.utils.logger import Logger
from dreamer.configs import (
    sys_config,
    analysis_config
)

import mpmath as mp
from tqdm import tqdm
from dreamer.utils.constants.constant import Constant


class Analyzer(AnalyzerScheme):
    def __init__(self, const: Constant, cmf: CMF, shards: List[Searchable]):
        self.cmf = cmf
        self.const = const
        self.shards = shards

    def search(self) -> Dict[Searchable, DataManager]:
        managers = {}

        for i, shard in enumerate((prog_bar := tqdm(self.shards, desc=f'Analyzing shards', **sys_config.TQDM_CONFIG))):
            # with Logger.simple_timer(f'get start point for shard'):
            start = shard.get_interior_point()
            Logger(f'{">" * 10} SHARD NO. {i + 1} {"<" * 10}').log(msg_prefix='\n', print=prog_bar.write)
            if analysis_config.SHOW_START_POINT:
                Logger(f'Chosen shard start point: {start}', Logger.Levels.info).log(print=prog_bar.write)
            if analysis_config.SHOW_SEARCHABLE:
                Logger(f'Shard: \n{shard}', Logger.Levels.info).log(print=prog_bar.write)

            # with Logger.simple_timer(f'preform search'):
            searcher = SerialSearcher(shard, self.const, use_LIReC=analysis_config.USE_LIReC)
            dm = searcher.search(
                start,
                find_limit=analysis_config.ANALYZE_LIMIT,
                find_gcd_slope=analysis_config.ANALYZE_GCD_SLOPE,
                find_eigen_values=analysis_config.ANALYZE_EIGEN_VALUES,
                trajectory_generator=analysis_config.NUM_TRAJECTORIES_FROM_DIM
            )

            identified = dm.identified_percentage
            best_delta = dm.best_delta[0]
            best_trajectory = dm.best_delta[1]

            if analysis_config.PRINT_FOR_EVERY_SEARCHABLE:
                if best_delta is None:
                    Logger(
                        f'Identified {identified * 100:.2f}% of trajectories as containing "{self.const.name}"'
                        f'best delta: {best_delta}\n\t [ at trajectory: {best_trajectory} ]',
                        Logger.Levels.info
                    ).log(print=prog_bar.write)
                else:
                    Logger(
                        f'Identified {identified * 100:.2f}% of trajectories as containing "{self.const.name}"'
                        f'best delta: {best_delta:.4f}\n\t[ at trajectory: {best_trajectory} ]'
                        f'\n\tp,q vectors: {dm[best_trajectory].initial_values.tolist()}',
                        Logger.Levels.info
                    ).log(print=prog_bar.write)
            if identified > analysis_config.IDENTIFY_THRESHOLD and best_delta is not None:
                managers[shard] = dm
            else:
                if best_delta is not None:
                    Logger(
                        f'Ignoring shard - identified <= {analysis_config.IDENTIFY_THRESHOLD * 100}% '
                        f'of tested trajectories',
                        Logger.Levels.info
                    ).log(print=prog_bar.write)
                else:
                    Logger(f'No best delta was found', Logger.Levels.warning).log(print=prog_bar.write)
        return managers

    def prioritize(self, managers: Dict[Searchable, DataManager], ranks=3) -> Dict[Searchable, Dict[str, int]]:
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
                'dim': self.cmf.dim()
            }
        return ranked
