from abc import ABC, abstractmethod
from dreamer.loading.config import *
from dreamer.utils.constants.constant import Constant
from dreamer.utils.types import *


class Formatter(ABC):
    """
    This class defines the bridge between a CMF in Ramanujan Tools and a JSON representation of it.
    This class is a registry of all formatters
    """
    registry: Dict[str, Type['Formatter']] = dict()

    def __init__(self, const: str | Constant):
        self.const = const.name if isinstance(const, Constant) else const

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Formatter.registry[cls.__name__] = cls

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

    def _to_json_obj(self) -> dict:
        return {'const': self.const}

    @classmethod
    @abstractmethod
    def _from_json_obj(cls, obj: dict | list) -> object:
        raise NotImplementedError

    @classmethod
    def fetch_from_registry(cls, name: str) -> Type['Formatter']:
        if name in cls.registry:
            return Formatter.registry[name]
        raise KeyError(f'{name} is not registered as a Formatter')

    def to_json_obj(self) -> Dict[str, Any]:
        if not issubclass(self.__class__, Formatter):
            raise TypeError(f'Not a Formatter subclass: {type(self)}')
        return {TYPE_ANNOTATE: self.__class__.__name__, DATA_ANNOTATE: self._to_json_obj()}

    @classmethod
    def from_json_obj(cls, src: dict):
        try:
            if src[TYPE_ANNOTATE] not in cls.registry:
                raise NotImplementedError(f'Not a Formatter subclass: {src[TYPE_ANNOTATE]}')
            return cls.registry[src[TYPE_ANNOTATE]]._from_json_obj(src[DATA_ANNOTATE])
        except AttributeError:
            raise NotImplementedError(f'constructor for {src[TYPE_ANNOTATE]} is not implemented.'
                                      f' Make sure that the name of the file is the same as the class')
        except TypeError:
            raise NotImplementedError(f'All formatters must inherit from {cls.__name__} and be in registry')
