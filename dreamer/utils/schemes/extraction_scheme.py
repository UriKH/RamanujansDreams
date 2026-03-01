from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.schemes.module import Module
from dreamer.utils.types import *
from dreamer.utils.constants.constant import Constant

from abc import abstractmethod, ABC


class ExtractionModScheme(Module):
    """
    A template for a general searchable extraction module which wraps extraction method(s).
    """

    def __init__(self,
                 cmf_data: Dict[Constant, List[ShiftCMF]],
                 name: Optional[str] = None, desc: Optional[str] = None, version: Optional[str] = None,
                 ):
        """
        :param cmf_data: Mapping from constants to a list of CMFs and their shifts in start point.
        :param name: Optional name of the module
        :param desc: Optional description
        :param version: Optional version of the module
        """
        super().__init__(name, desc, version)
        self.cmf_data = cmf_data

    @abstractmethod
    def execute(self) -> Optional[Dict[Constant, List[Searchable]]]:
        """
        Preform extraction and return the results
        :return: A mapping from constants to a list of searchables.
        """
        raise NotImplementedError


class ExtractionScheme(ABC):
    """
    A template for a general extraction method.
    """

    def __init__(self, const: Constant, cmf_data: ShiftCMF):
        """
        :param const: A constant to extract searchables for.
        :param cmf_data: The CMF to extract searchables from and extra data about the CMF required for extraction.
        """
        self.const = const
        self.cmf_data: ShiftCMF = cmf_data

    @abstractmethod
    def extract_searchables(self, *args) -> List[Searchable]:
        """
        Extracts searchables from the CMF matching the provided shift.
        :return: A list of searchables found in the CMF.
        """
        raise NotImplementedError
    