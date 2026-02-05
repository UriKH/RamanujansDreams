from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.storage.exporter import Exporter, Formats
from dreamer.utils.storage.storage_objects import DataManager
from dreamer.utils.schemes.searcher_scheme import SearcherModScheme
from dreamer.utils.schemes.module import CatchErrorInModule
from dreamer.utils.types import *
from dreamer.utils.ui.tqdm_config import SmartTQDM
from ...methods.serial.serial_searcher import SerialSearcher
from . import config as search_config_local
from dreamer.configs import config
import os

search_config = config.search
sys_config = config.system


class SearcherModV1(SearcherModScheme):
    """
    A searcher module that performs a serial search over a list of searchable spaces.
    """

    def __init__(self, searchables: List[Searchable], use_LIReC: bool):
        """
        :param searchables: A list of searchable spaces to search in.
        :param use_LIReC: If true, LIReC will be used to identify constants within the searchable spaces.
        """
        super().__init__(
            searchables,
            use_LIReC,
            description='Searcher module - orchestrating a deep search within a prioritized list of spaces',
            version='0.0.1'
        )

    @CatchErrorInModule(with_trace=sys_config.MODULE_ERROR_SHOW_TRACE, fatal=True)
    def execute(self) -> Dict[Searchable, DataManager]:
        """
        Executes the search. Computes the results per searchable space and exports them into a file while running.
        :return: A mapping from searchables to their search results.
        """
        if not self.searchables:
            return dict()

        os.makedirs(
            dir_path := os.path.join(sys_config.EXPORT_SEARCH_RESULTS, self.searchables[0].const.name),
            exist_ok=True
        )

        dms: Dict[Searchable, DataManager] = dict()

        with Exporter.export_stream(dir_path, exists_ok=True, clean_exists=True, fmt=Formats.PICKLE) as write_chunk:
            for space in SmartTQDM(self.searchables, desc='Searching the searchable spaces: ', **sys_config.TQDM_CONFIG):
                searcher = SerialSearcher(space, space.const, use_LIReC=self.use_LIReC)
                res = searcher.search(
                    None,
                    find_limit=search_config_local.FIND_LIMIT,
                    find_gcd_slope=search_config_local.FIND_GCD_SLOPE,
                    find_eigen_values=search_config_local.FIND_EIGEN_VALUES,
                    trajectory_generator=search_config.NUM_TRAJECTORIES_FROM_DIM
                )
                write_chunk(res)

        return dms
