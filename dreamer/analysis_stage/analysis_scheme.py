from .shards.searchable import Searchable
from dreamer.search_stage.data_manager import DataManager
from dreamer.system.module import Module
from dreamer.utils.types import *

from abc import abstractmethod, ABC


class AnalyzerModScheme(Module):
    @abstractmethod
    def execute(self) -> Dict[str, List[Searchable]]:
        raise NotImplementedError


class AnalyzerScheme(ABC):
    @abstractmethod
    def search(self) -> Dict[Searchable, DataManager]:
        raise NotImplementedError

    @abstractmethod
    def prioritize(self, managers: Dict[Searchable, DataManager], ranks: int) -> Dict[Searchable, Dict[str, int]]:
        raise NotImplementedError
