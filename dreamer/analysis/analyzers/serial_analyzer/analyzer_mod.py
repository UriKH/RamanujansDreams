from dreamer.utils.schemes.analysis_scheme import AnalyzerModScheme
from dreamer.utils.ui.tqdm_config import SmartTQDM
from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.logger import Logger
from dreamer.utils.types import *
from dreamer.utils.schemes.module import CatchErrorInModule
from dreamer.utils.constants.constant import Constant
from dreamer.configs import sys_config
from dreamer.analysis.analysis_methods.serial_analyzer import Analyzer
from .config import *


class AnalyzerModV1(AnalyzerModScheme):
    """
    The class represents the module for CMF analysis and shard search filtering and prioritization.
    """

    def __init__(self, cmf_data: Dict[Constant, List[Searchable]]):
        """
        :param cmf_data: A mapping from each constant to the list of searchables.
        """
        super().__init__(
            cmf_data,
            desc='Module for CMF analysis and shard search filtering and prioritization',
            version='1'
        )

    @CatchErrorInModule(with_trace=sys_config.MODULE_ERROR_SHOW_TRACE, fatal=True)
    def execute(self) -> Dict[Constant, List[Searchable]]:
        """
        Preforms an analysis using an analyzer and prioritize using its suggestions.
        :return: A mapping from each constant to a list of prioritized searchables for deeper search.
        """
        def merge_dicts(dict_list: List[Dict]) -> Dict:
            merged = {}
            for d in dict_list:
                merged.update(d)
            return merged

        queues = {c: [] for c in self.cmf_data.keys()}
        for constant, shards in SmartTQDM(self.cmf_data.items(),
                                          desc='Analyzing constants and their CMFs', **sys_config.TQDM_CONFIG):
            queue: List[Dict[Searchable, Dict[str, int]]] = []
            Logger(
                Logger.buffer_print(sys_config.LOGGING_BUFFER_SIZE, f'Analyzing for {constant.name}', '=')
            ).log(msg_prefix='\n')

            # TODO: add option to use mpf - depends on the use_LIReC I guess. maybe there is a way to use only sympy format
            analyzer = Analyzer(constant, shards)
            dms = analyzer.search()
            queue.append(analyzer.prioritize(dms, PRIORITIZATION_RANKS))
            # TODO: Now we want to take the DataManagers and convert whose to databases per CMF - I don't know if we really want this or not...
            merged: Dict[Searchable, Dict[str, int]] = merge_dicts(queue)
            queues[constant] = sorted(
                merged.keys(),
                key=lambda space: (-merged[space]['delta_rank'], merged[space]['dim'])
            )
        return queues
