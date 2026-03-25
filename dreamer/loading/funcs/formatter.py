from abc import ABC, abstractmethod
from dreamer.loading.config import *
from dreamer.utils.constants.constant import Constant
from dreamer.utils.types import *
from dreamer.configs import config
import json


class Formatter(ABC):
    """
    This class defines the bridge between a CMF in Ramanujan Tools and a JSON representation of it.
    This class is a registry of all formatters
    """
    registry: Dict[str, Type['Formatter']] = dict()

    def __init__(self, const: str | Constant, shifts: Optional[list] = None,
                 selected_start_points: Optional[List[Tuple[Union[int, sp.Rational], ...]]] = None,
                 only_selected: bool = False,
                 use_inv_t: bool = None):
        if use_inv_t is None:
            use_inv_t = config.search.DEFAULT_USES_INV_T

        self.const = const.name if isinstance(const, Constant) else const
        self.shifts = shifts
        self.selected_start_points = selected_start_points
        self.only_selected = only_selected
        self.use_inv_t = use_inv_t

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Formatter.registry[cls.__name__] = cls

    @abstractmethod
    def __repr__(self):
        return json.dumps(self._to_json_obj())

    @abstractmethod
    def __str__(self):
        return f'<{self.__class__.__name__}: {self.__repr__()}>'

    @abstractmethod
    def __hash__(self):
        return hash((
            self.const, tuple(self.shifts), frozenset(self.selected_start_points),
            self.only_selected, self.use_inv_t
        ))

    @abstractmethod
    def to_cmf(self) -> ShiftCMF:
        """
        Converts the Formatter to a CMF.
        :return: The CMF with shift as ShiftCMF object
        """
        raise NotImplementedError

    def _to_json_obj(self) -> dict:
        # Prepare shifts
        shifts = self.shifts
        if shifts:
            shifts = [str(shift) if isinstance(shift, sp.Expr) else shift for shift in self.shifts]

        # Prepare start points
        points = self.selected_start_points
        if points:
            points = [[v if isinstance(v, int) else str(v) for v in p] for p in self.selected_start_points]

        return {
            'const': self.const.name if isinstance(self.const, Constant) else self.const,
            'use_inv_t': self.use_inv_t,
            'shifts': shifts,
            'selected_start_points': points,
            'only_selected': self.only_selected
        }

    @classmethod
    @abstractmethod
    def _from_json_obj(cls, obj: dict | list) -> object:
        raise NotImplementedError

    @staticmethod
    def _shift_from_json(data):
        return [sp.sympify(shift) if isinstance(shift, str) else shift for shift in data]

    @staticmethod
    def _selected_start_points_from_json(data):
        points = []
        for point_list in data:
            points.append(tuple(sp.sympify(v) if isinstance(v, str) else v for v in point_list))
        return points

    @classmethod
    def fetch_from_registry(cls, name: str) -> Type['Formatter']:
        """
        Checks if a formatter is registered and returns it
        :param name: The name of the formatter class
        :return: The formatter class
        """
        if name in cls.registry:
            return Formatter.registry[name]
        raise KeyError(f'{name} is not registered as a Formatter')

    def to_json_obj(self) -> Dict[str, Any]:
        """
        Converts the Formatter to a JSON object
        :return: The JSON like object (dictionary)
        """
        if not issubclass(self.__class__, Formatter):
            raise TypeError(f'Not a Formatter subclass: {type(self)}')
        return {TYPE_ANNOTATE: self.__class__.__name__, DATA_ANNOTATE: self._to_json_obj()}

    @classmethod
    def from_json_obj(cls, src: dict) -> 'Formatter':
        """
        Converts from a JSON object to the relevant Formatter
        :param src: The source JSON like object (dictionary)
        :return: The Formatter object
        """
        try:
            if src[TYPE_ANNOTATE] not in cls.registry:
                raise NotImplementedError(f'Not a Formatter subclass: {src[TYPE_ANNOTATE]}')
            return cls.registry[src[TYPE_ANNOTATE]]._from_json_obj(src[DATA_ANNOTATE])
        except AttributeError:
            raise NotImplementedError(f'constructor for {src[TYPE_ANNOTATE]} is not implemented.'
                                      f' Make sure that the name of the file is the same as the class')
        except TypeError:
            raise NotImplementedError(f'All formatters must inherit from {cls.__name__} and be in registry')
