from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.storage.storage_objects import DataManager
from dreamer.utils.schemes.module import Module
from dreamer.utils.types import *
from dreamer.utils.constants.constant import Constant

from abc import abstractmethod, ABC


class AnalyzerModScheme(Module):
    """
    A template for a general analysis module which wraps analysis method(s).
    """
    def __init__(self,
                 cmf_data: Dict[Constant, List[Searchable]],
                 name: Optional[str] = None, desc: Optional[str] = None, version: Optional[str] = None
                 ):
        """
        :param cmf_data: Mapping from constants to a list of searchables.
        :param name: Optional name of the module
        :param desc: Optional description
        :param version: Optional version of the module
        """
        super().__init__(name, desc, version)
        self.cmf_data = cmf_data

    @abstractmethod
    def execute(self) -> Dict[Constant, List[Searchable]]:
        """
        Preform analysis and return the results
        :return: A mapping from constants to a list of prioritized searchables
        """
        raise NotImplementedError


class AnalyzerScheme(ABC):
    """
    A template for a general analyzer.
    """

    @abstractmethod
    def search(self) -> Dict[Searchable, DataManager]:
        """
        Preform search and return the results (meant to be used by the execute method in AnalyzerModScheme)
        :return: A mapping from searchables to the search result.
        """
        raise NotImplementedError

    @abstractmethod
    def prioritize(self, managers: Dict[Searchable, DataManager], *args) -> Dict[Searchable, Any]:
        """
        Given a set of searchables and their search results, prioritize them.
        :param managers: A mapping from a searchable to its search result.
        :return: The mapping from searchables to their prioritization object (e.g., a score or multiple scores).
        """
        raise NotImplementedError
