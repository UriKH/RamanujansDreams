from dreamer.utils.schemes.searcher_scheme import SearchMethod, SearcherModScheme
from dreamer.utils.storage.storage_objects import DataManager
from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.types import *   # for using all the typechecking stuff
from dreamer.utils.schemes.module import CatchErrorInModule
from dreamer.utils.constants.constant import Constant
from dreamer.configs.system import sys_config
import os
from tqdm import tqdm
from dreamer.utils.storage.exporter import Exporter, Formats
from dreamer.utils.constant_transform import get_const_as_sp


class MySearchMethod(SearchMethod):
    def __init__(self,
                 space: Searchable,
                 constant,  # sympy constant or mp.mpf
                 # TODO: <your arguments here>
                 data_manager: DataManager = None,
                 share_data: bool = True,
                 use_LIReC: bool = True):
        super().__init__(space, constant, use_LIReC, data_manager, share_data)
        # TODO: set arguments

    def search(self) -> DataManager:
        # TODO: compute and search in the space. When finished - return your data manager
        return self.data_manager


class MySearchMod(SearcherModScheme):
    def __init__(
            self,
            searchables: List[Searchable],
            use_LIReC: Optional[bool] = True,
            # TODO: <your arguments here>
            #  Note: that if you add more arguments you would probably need to use functools.partial in System().
            #  This allows you to add as many arguments as you want without any need to change System's impelementation.
    ):
        super().__init__(
            searchables, use_LIReC,
            name='A very witty name',
            description='My super cool and smart module using super smart methods beyond your comprehension',
            version='your version here :)'
        )
        # TODO: set arguments

    @CatchErrorInModule(with_trace=sys_config.MODULE_ERROR_SHOW_TRACE, fatal=True)
    def execute(self):
        # Create a folder for the stored results created by the searcher
        if self.searchables:
            os.makedirs(
                dir_path := os.path.join(sys_config.EXPORT_SEARCH_RESULTS, self.searchables[0].const_name),
                exist_ok=True
            )
        else:
            dir_path = sys_config.EXPORT_SEARCH_RESULTS

        # This bit of code does the following:
        # When you are searching per searchable (e.g. Shard) the results are stored in automatically in a pickle file
        # in `dir_path`
        with Exporter.export_stream(dir_path, exists_ok=True, clean_exists=True, fmt=Formats.PICKLE) as write_chunk:
            for space in tqdm(self.searchables, desc='Searching the searchable spaces: ', **sys_config.TQDM_CONFIG):
                searcher = MySearchMethod(
                    space, Constant.get_constant(space.const_name)    # TODO: add your arguments
                )   # creates an instance of your searcher
                res = searcher.search(
                    # TODO: all you searcher's arguments here
                )
                write_chunk(res)    # Writes the search results into a file
