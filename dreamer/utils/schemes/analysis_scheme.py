from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.storage.storage_objects import DataManager
from dreamer.utils.schemes.module import Module
from dreamer.utils.types import *
from dreamer.utils.constants.constant import Constant

from abc import abstractmethod, ABC


class AnalyzerModScheme(Module):
    @abstractmethod
    def execute(self) -> Dict[str, List[Searchable]]:
        raise NotImplementedError

    # @abstractmethod
    # def get_searchables(self) -> Dict[Constant, List[Searchable]]:
    #     raise NotImplementedError


class AnalyzerScheme(ABC):
    @abstractmethod
    def search(self) -> Dict[Searchable, DataManager]:
        raise NotImplementedError

    @abstractmethod
    def prioritize(self, managers: Dict[Searchable, DataManager], ranks: int) -> Dict[Searchable, Dict[str, int]]:
        raise NotImplementedError

    # @abstractmethod
    # def get_searchables(self) -> List[Searchable]:
    #     raise NotImplementedError
