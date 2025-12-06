from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.storage import Exporter, Formats
from dreamer.utils.storage.storage_objects import DataManager
from dreamer.utils.schemes.searcher_scheme import SearcherModScheme
from ...methods.serial.serial_searcher import SerialSearcher
from . import config as search_config_local
from dreamer.configs.search import search_config
from dreamer.utils.types import *
from dreamer.utils.schemes.module import CatchErrorInModule
from dreamer.utils.constant_transform import *
from dreamer.utils.constants.constant import Constant


from tqdm import tqdm
import os


class SearcherModV1(SearcherModScheme):
    """

    """

    def __init__(self, searchables: List[Searchable], use_LIReC: bool):
        super().__init__(
            searchables,
            use_LIReC,
            description='Searcher module - orchestrating a deep search within a prioritized list of spaces',
            version='0.0.1'
        )

    @CatchErrorInModule(with_trace=sys_config.MODULE_ERROR_SHOW_TRACE, fatal=True)
    def execute(self) -> Dict[Searchable, DataManager]:
        # create folder
        if self.searchables:
            os.makedirs(
                dir_path := os.path.join(sys_config.EXPORT_SEARCH_RESULTS, self.searchables[0].const_name),
                exist_ok=True
            )
        else:
            dir_path = sys_config.EXPORT_SEARCH_RESULTS

        dms: Dict[Searchable, DataManager] = dict()

        with Exporter.export_stream(dir_path, exists_ok=True, clean_exists=True, fmt=Formats.PICKLE) as write_chunk:
            for space in tqdm(self.searchables, desc='Searching the searchable spaces: ', **sys_config.TQDM_CONFIG):
                searcher = SerialSearcher(space, Constant.get_constant(space.const_name), use_LIReC=self.use_LIReC)
                res = searcher.search(
                    None,
                    find_limit=search_config_local.FIND_LIMIT,
                    find_gcd_slope=search_config_local.FIND_GCD_SLOPE,
                    find_eigen_values=search_config_local.FIND_EIGEN_VALUES,
                    trajectory_generator=search_config.NUM_TRAJECTORIES_FROM_DIM
                )
                write_chunk(res)

        return dms
