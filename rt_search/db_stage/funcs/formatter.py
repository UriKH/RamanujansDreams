from abc import ABC, abstractmethod
import json

from rt_search.utils.types import *
from rt_search.db_stage.config import *

from . import FORMATTER_REGISTRY

from ...utils.types import *


@dataclass
class Formatter(ABC):
    const: str

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def __hash__(self):
        raise NotImplementedError

    @abstractmethod
    def to_cmf(self) -> ShiftCMF:
        raise NotImplementedError

    @classmethod
    def get_type_name(cls) -> str:
        return cls.__class__.__name__

    def _to_json_obj(self) -> dict:
        return {'const': self.const}

    @classmethod
    @abstractmethod
    def _from_json_obj(cls, obj: dict | list) -> object:
        raise NotImplementedError

    def to_json_obj(self) -> Dict[str, Any]:
        if not issubclass(self.__class__, Formatter):
            raise TypeError(f'Not a Formatter subclass: {type(self)}')
        return {TYPE_ANNOTATE: self.get_type_name(), DATA_ANNOTATE: self._to_json_obj()}

    @classmethod
    def from_json_obj(cls, src: dict):
        try:
            if (src[TYPE_ANNOTATE] not in FORMATTER_REGISTRY or
                    not issubclass(t := FORMATTER_REGISTRY[src[TYPE_ANNOTATE]], Formatter)):
                raise TypeError
            return t._from_json_obj(json.dumps(src[DATA_ANNOTATE]))
        except AttributeError:
            raise NotImplementedError(f'constructor for {src[TYPE_ANNOTATE]} is not implemented.'
                                      f' Make sure that the name of the file is the same as the class')
        except TypeError:
            raise NotImplementedError(f'All formatters must inherit from {cls.__name__} and be in registry')
