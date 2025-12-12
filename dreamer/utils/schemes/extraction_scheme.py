from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.storage.storage_objects import DataManager
from dreamer.utils.schemes.module import Module
from dreamer.utils.types import *
from dreamer.utils.constants.constant import Constant

from abc import abstractmethod, ABC


class ExtractionModScheme(Module):
    def __init__(self,
                 cmf_data: Dict[Constant, List[ShiftCMF]],
                 name: Optional[str] = None, desc: Optional[str] = None, version: Optional[str] = None
                 ):
        super().__init__(name, desc, version)
        self.cmf_data = cmf_data

    @abstractmethod
    def execute(self) -> Optional[Dict[Constant, List[Searchable]]]:
        raise NotImplementedError


class ExtractionScheme(ABC):
    def __init__(self, const: Constant, cmf: CMF, shift: Position):
        self.const = const
        self.cmf: CMF = cmf
        self.shift: Position = shift

    @abstractmethod
    def extract_searchables(self) -> List[Searchable]:
        raise NotImplementedError
    