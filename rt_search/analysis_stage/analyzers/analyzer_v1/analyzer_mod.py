from ...analysis_scheme import AnalyzerModScheme
from .analyzer import Analyzer
from ...subspaces.searchable import Searchable
from rt_search.utils.geometry.point_generator import PointGenerator
from rt_search.utils.logger import Logger
from rt_search.utils.types import *
from rt_search.system.system import System
from rt_search.system.module import CatchErrorInModule
from rt_search.configs import (
    sys_config,
    analysis_config
)
from .config import *

from tqdm import tqdm
from dataclasses import astuple


class AnalyzerModV1(AnalyzerModScheme):
    """
    The class represents the module for CMF analysis and shard search filtering and prioritization.
    """

    def __init__(self, cmf_data: Dict[str, List[ShiftCMF]]):
        super().__init__(
            description='Module for CMF analysis and shard search filtering and prioritization',
            version='1'
        )
        self.cmf_data = cmf_data

    @CatchErrorInModule(with_trace=sys_config.MODULE_ERROR_SHOW_TRACE, fatal=True)
    def execute(self) -> Dict[str, List[Searchable]]:
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
        for constant, cmf_tups in tqdm(self.cmf_data.items(), desc='Analyzing constants and their CMFs',
                                       **sys_config.TQDM_CONFIG):
            queue: List[Dict[Searchable, Dict[str, int]]] = []

            Logger(
                Logger.buffer_print(sys_config.LOGGING_BUFFER_SIZE, f'Analyzing for {constant}', '=')
            ).log(msg_prefix='\n')
            for t in cmf_tups:
                Logger(
                    Logger.buffer_print(sys_config.LOGGING_BUFFER_SIZE, f'Current CMF: {t.cmf} with shift {t.shift}', '=')
                ).log(msg_prefix='\n')
                # TODO: add option to use mpf - depends on the use_LIReC I guess. maybe there is a way to use only sympy format
                analyzer = Analyzer(constant, t.cmf, t.shift, System.get_const_as_sp(constant))
                dim = t.cmf.dim()
                dms = analyzer.search(
                    length=PointGenerator.calc_sphere_radius(analysis_config.NUM_TRAJECTORIES_FROM_DIM(dim), dim)
                )
                queue.append(analyzer.prioritize(dms, PRIORITIZATION_RANKS))
                # TODO: Now we want to take the DataManagers and convert whose to databases per CMF - I don't know if we really want this or not...
            merged: Dict[Searchable, Dict[str, int]] = merge_dicts(queue)
            queues[constant] = sorted(
                merged.keys(),
                key=lambda space: (-merged[space]['delta_rank'], merged[space]['dim'])
            )
        return queues
