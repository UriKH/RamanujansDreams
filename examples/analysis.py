from dreamer.utils.schemes.analysis_scheme import AnalyzerScheme, AnalyzerModScheme
from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.schemes.module import CatchErrorInModule
from dreamer.utils.constants.constant import Constant
from dreamer.utils.storage.storage_objects import DataManager
from dreamer.utils.types import *
from dreamer.utils.logger import Logger
from dreamer.configs import sys_config
from dreamer.utils.ui.tqdm_config import SmartTQDM


class MyAnalyzer(AnalyzerScheme):
    def __init__(
            self,
            const: Constant,
            cmf: CMF,
            shards: List[Searchable],
            # TODO: <your arguments here>
    ):
        self.cmf = cmf
        self.const = const
        self.shards = shards

    def search(self) -> Dict[Searchable, DataManager]:
        # TODO: compute and search in the space. When finished - return your mapping of Searcgabke to DataManager
        pass

    def prioritize(
            self,
            managers: Dict[Searchable, DataManager],
            ranks=3     # This argument could be ignored as the module only wraps the analyzer
                        # - thus you can do as you wish :)
            # TODO: <your arguments here>
    ) -> Dict[Searchable, Dict[str, int]]:
        """
        Example: The idea here is to prioritize searchables by the delta found in them and by the dimension of the space.
        We need not only a rank (delta rank) but also a dimension as we would like to search the lowest dimensional spaces
         before searching in big ones.
        """
        # TODO: from the given a mapping prioritize the searchables via: ('delta_rank' - int, 'dim' - int)
        #   The above is a standard created for the simple analyzer module. You are free to use whichever prioritization
        #   method you prefer and use it in your module below.
        pass


class MyAnalyzerMod(AnalyzerModScheme):
    """
    The class represents the module for CMF analysis and shard search filtering and prioritization.
    """

    def __init__(self, cmf_data: Dict[Constant, List[Searchable]]):
        super().__init__(
            cmf_data,
            name='A very witty name',
            description='My super cool and smart module using super smart methods beyond your comprehension',
            version='your version here :)'
        )
        # TODO: set arguments

    @CatchErrorInModule(with_trace=sys_config.MODULE_ERROR_SHOW_TRACE, fatal=True)
    def execute(self) -> Dict[Constant, List[Searchable]]:
        """
        TODO:
            The main function of the module.
            Here you probably just want to call your analyzer and prioritizer.
            The output is a mapping from constant to a list of prioritized searchables.
            The following is a template you do not have to follow but might be helpful
        """

        queues = {c: [] for c in self.cmf_data.keys()}
        for constant, shards in SmartTQDM(
                self.cmf_data.items(), desc='Analyzing constants and their CMFs', **sys_config.TQDM_CONFIG
        ):
            # These sleeps are just to make the output look nicer (these could be ignored using the configurations)
            Logger.sleep(1)
            analyzer = MyAnalyzer(constant, shards[0].cmf, shards)
            Logger.sleep(1)

            dms = analyzer.search()
            prioritization: Dict[Searchable, Dict[str, int]] = analyzer.prioritize(
                dms,
                # all your other arguments...
            )

            # TODO: update 'queues[constant]' according to the 'prioritization'
        return queues
