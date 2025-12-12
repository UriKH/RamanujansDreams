from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.storage.storage_objects import DataManager
from dreamer.utils.schemes.module import Module
from dreamer.utils.types import *
from dreamer.utils.constants.constant import Constant

from abc import abstractmethod, ABC


class AnalyzerModScheme(Module):
    def __init__(self,
                 cmf_data: Dict[Constant, List[Searchable]],
                 name: Optional[str] = None, desc: Optional[str] = None, version: Optional[str] = None
                 ):
        super().__init__(name, desc, version)
        self.cmf_data = cmf_data

    @abstractmethod
    def execute(self) -> Dict[Constant, List[Searchable]]:
        raise NotImplementedError


class AnalyzerScheme(ABC):
    @abstractmethod
    def search(self) -> Dict[Searchable, DataManager]:
        raise NotImplementedError

    @abstractmethod
    def prioritize(self, managers: Dict[Searchable, DataManager], ranks: int) -> Dict[Searchable, Dict[Constant, int]]:
        raise NotImplementedError
