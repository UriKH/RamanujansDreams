import os
import time

from dreamer.utils.schemes.analysis_scheme import AnalyzerModScheme
from .analyzer import Analyzer
from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.logger import Logger
from dreamer.utils.types import *
from dreamer.utils.schemes.module import CatchErrorInModule
from dreamer.utils.constants.constant import Constant
from dreamer.configs import sys_config
from .config import *
from tqdm import tqdm


class AnalyzerModV1(AnalyzerModScheme):
    """
    The class represents the module for CMF analysis and shard search filtering and prioritization.
    """

    def __init__(self, cmf_data: Dict[Constant, List[Searchable]]):
        super().__init__(
            cmf_data,
            desc='Module for CMF analysis and shard search filtering and prioritization',
            version='1'
        )

    # def _create_analyzers(self) -> List[Analyzer]:
    #     def merge_dicts(dict_list: List[Dict]) -> Dict:
    #         merged = {}
    #         for d in dict_list:
    #             merged.update(d)
    #         return merged
    #
    #     queues = {c: [] for c in self.cmf_data.keys()}
    #     for constant, cmf_tups in tqdm(self.cmf_data.items(), desc='Analyzing constants and their CMFs',
    #                                    **sys_config.TQDM_CONFIG):
    #         queue: List[Dict[Searchable, Dict[str, int]]] = []
    #
    #         Logger(
    #             Logger.buffer_print(sys_config.LOGGING_BUFFER_SIZE, f'Analyzing for {constant.name}', '=')
    #         ).log(msg_prefix='\n')
    #         for t in cmf_tups:
    #             if t.raw:
    #                 Logger(
    #                     Logger.buffer_print(sys_config.LOGGING_BUFFER_SIZE,
    #                                         f'Current CMF is manual with dim={t.cmf.dim()} and shift {t.shift}', '=')
    #                 ).log(msg_prefix='\n')
    #             else:
    #                 Logger(
    #                     Logger.buffer_print(sys_config.LOGGING_BUFFER_SIZE, f'Current CMF: {t.cmf} with shift {t.shift}', '=')
    #                 ).log(msg_prefix='\n')
    #             analyzer = Analyzer(constant, t.cmf, t.shift, constant)
    #
    # def get_searchables(self) -> Dict[Constant, List[Searchable]]:
    #     pass

    @CatchErrorInModule(with_trace=sys_config.MODULE_ERROR_SHOW_TRACE, fatal=True)
    def execute(self) -> Dict[Constant, List[Searchable]]:
        """
        The main function of the module. It performs the following steps:
        * Store all CMFs
        * extract shards for each CMF
        * for each CMF generate start points and trajectories
        * search each shard for shallow search and get the data
        * prioritize for deep search
        """
        def merge_dicts(dict_list: List[Dict]) -> Dict:
            merged = {}
            for d in dict_list:
                merged.update(d)
            return merged

        queues = {c: [] for c in self.cmf_data.keys()}
        for constant, shards in tqdm(self.cmf_data.items(), desc='Analyzing constants and their CMFs',
                                     **sys_config.TQDM_CONFIG):
            queue: List[Dict[Searchable, Dict[str, int]]] = []

            Logger.sleep(1)
            Logger(
                Logger.buffer_print(sys_config.LOGGING_BUFFER_SIZE, f'Analyzing for {constant.name}', '=')
            ).log(msg_prefix='\n')
            # for s in shards:
            cmf = shards[0].cmf
            if issubclass(cmf.__class__, CMF):
                Logger(
                    Logger.buffer_print(sys_config.LOGGING_BUFFER_SIZE,
                                        f'Current CMF is manual with dim={cmf.dim()} and shift {shards[0].shift}', '=')
                ).log(msg_prefix='\n')
            else:
                Logger(
                    Logger.buffer_print(sys_config.LOGGING_BUFFER_SIZE, f'Current CMF: {cmf} with shift {shards[0].shift}', '=')
                ).log(msg_prefix='\n')
                # TODO: add option to use mpf - depends on the use_LIReC I guess. maybe there is a way to use only sympy format
            Logger.sleep(1)
            analyzer = Analyzer(constant, shards[0].cmf, shards)
            Logger.sleep(1)
            dms = analyzer.search()
            queue.append(analyzer.prioritize(dms, PRIORITIZATION_RANKS))
                # TODO: Now we want to take the DataManagers and convert whose to databases per CMF - I don't know if we really want this or not...
            merged: Dict[Searchable, Dict[str, int]] = merge_dicts(queue)
            queues[constant] = sorted(
                merged.keys(),
                key=lambda space: (-merged[space]['delta_rank'], merged[space]['dim'])
            )
        return queues
