from rt_search.analysis_stage.shards.searchable import Searchable
from ...data_manager import DataManager
from ...searcher_scheme import SearcherModScheme
from ...methods.serial.serial_searcher import SerialSearcher
from . import config as search_config
from rt_search.utils.types import *
from rt_search.system.system import System
from rt_search.system.module import CatchErrorInModule
from rt_search.configs import sys_config

from tqdm import tqdm
# import queue
import os
import json


class SearcherModV1(SearcherModScheme):
    """

    """

    def __init__(self, searchables: List[Searchable], use_LIReC: bool):
        super().__init__(
            description='Searcher module - orchestrating a deep search within a prioritized list of spaces',
            version='0.0.1'
        )
        self.searchables = searchables
        self.use_LIReC = use_LIReC

    @CatchErrorInModule(with_trace=sys_config.MODULE_ERROR_SHOW_TRACE, fatal=True)
    def execute(self) -> Dict[Searchable, DataManager]:
        # q = queue.Queue()
        # writer = DBWriter('./my_results.db', q, 1000)
        # writer.start()

        # create folder
        if self.searchables:
            os.makedirs(
                dir_path := os.path.join(sys_config.EXPORT_SEARCH_RESULTS, self.searchables[0].const_name),
                exist_ok=True
            )

        dms: Dict[Searchable, DataManager] = dict()
        it = 1
        for space in tqdm(self.searchables, desc='Searching the searchable spaces: ', **sys_config.TQDM_CONFIG):
            searcher = SerialSearcher(space, System.get_const_as_sp(space.const_name), use_LIReC=self.use_LIReC)
            res = searcher.search(
                None, partial_search_factor=1,
                find_limit=search_config.FIND_LIMIT,
                find_gcd_slope=search_config.FIND_GCD_SLOPE,
                find_eigen_values=search_config.FIND_EIGEN_VALUES
            )
            path = os.path.join(dir_path, f'space_{it}.json')
            with open(path, 'w') as f:
                json.dump({'space': space.to_json_obj(), 'result': res.to_json_obj()}, f, indent=2)
            it += 1
            # TODO: !!!!!!!!!!!!!!!!!!!!!!

        return dms
