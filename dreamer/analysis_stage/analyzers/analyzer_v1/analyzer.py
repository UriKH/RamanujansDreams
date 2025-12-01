from ...analysis_scheme import AnalyzerScheme
from ...shards.searchable import Searchable
from ...shards.extractor import ShardExtractor
from dreamer.search_stage.data_manager import DataManager
from dreamer.search_stage.methods.serial.serial_searcher import SerialSearcher
from dreamer.utils.types import *
from dreamer.utils.logger import Logger
from dreamer.configs import (
    sys_config,
    analysis_config
)

import mpmath as mp
from tqdm import tqdm


class Analyzer(AnalyzerScheme):
    def __init__(self, const_name: str, cmf: CMF, shift: Position, constant: mp.mpf):
        self.cmf = cmf
        self.shift = shift
        self.constant = constant
        self.extractor = ShardExtractor(const_name, cmf, shift)
        self.shards = self.extractor.extract_shards()

    def search(self) -> Dict[Searchable, DataManager]:
        managers = {}

        for shard in tqdm(self.shards, desc=f'Analyzing shards', **sys_config.TQDM_CONFIG):
            start = shard.get_interior_point()

            searcher = SerialSearcher(shard, self.constant, use_LIReC=analysis_config.USE_LIReC, deep_search=False)
            searcher.generate_trajectories(analysis_config.NUM_TRAJECTORIES_FROM_DIM(len(shard.symbols)))
            dm = searcher.search(
                start,
                partial_search_factor=analysis_config.PARTIAL_SEARCH_FACTOR,
                find_limit=analysis_config.ANALYZE_LIMIT,
                find_gcd_slope=analysis_config.ANALYZE_GCD_SLOPE,
                find_eigen_values=analysis_config.ANALYZE_EIGEN_VALUES
            )

            identified = dm.identified_percentage
            best_delta = dm.best_delta[0]
            if analysis_config.PRINT_FOR_EVERY_SEARCHABLE:
                if best_delta is None:
                    Logger(f'Identified {identified * 100:.2f}% of trajectories, best delta: {best_delta}',
                           Logger.Levels.info).log(msg_prefix='\n')
                else:
                    Logger(f'Identified {identified * 100:.2f}% of trajectories, best delta: {best_delta:.4f}',
                           Logger.Levels.info).log(msg_prefix='\n')
            if identified > analysis_config.IDENTIFY_THRESHOLD and best_delta is not None:
                managers[shard] = dm
            else:
                if best_delta is not None:
                    Logger(
                        f'Ignoring shard - identified <= {analysis_config.IDENTIFY_THRESHOLD ** 100}% '
                        f'of tested trajectories',
                        Logger.Levels.info
                    ).log(msg_prefix='\n')
                else:
                    Logger(f'No best delta was found', Logger.Levels.info).log()
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
