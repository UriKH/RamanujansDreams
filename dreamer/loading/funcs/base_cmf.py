import json

from ramanujantools import Matrix
from dreamer.utils.constants.constant import Constant
from dreamer.loading.funcs.formatter import Formatter
from dreamer.utils.types import *


class BaseCMF(Formatter):
    def __init__(self,
                 const: str | Constant, cmf: CMF, shifts: Optional[list] = None,
                 selected_start_points: Optional[List[Tuple[Union[int, sp.Rational], ...]]] = None,
                 only_selected: bool = False,
                 use_inv_t: bool = None
                 ):
        """
        Represents a general CMF and allows conversion to and from JSON.
        :var const: The constant related to this CMF
        :var cmf: The CMF.
        :var shifts: The shifts in starting point in the CMF where a sp.Rational indicates a shift.
        While 0 indicates no shift (None if not doesn't matter).
        :param selected_start_points: Optional list of start points to extract shards from.
        :param only_selected: If True, only extract shards from the selected start points.
        """
        super().__init__(const, shifts, selected_start_points, only_selected, use_inv_t)

        self.cmf = cmf
        if self.shifts is None:
            self.shifts = [0] * self.cmf.dim()

        if not isinstance(self.shifts, list) and not isinstance(self.shifts, Position):
            raise ValueError("Shifts should be a list or Position")

    @classmethod
    def _from_json_obj(cls, data: dict | list) -> "BaseCMF":
        """
        Converts a JSON string to a BaseCMF.
        :param data: The JSON string representation of the pFq (only attributes).
        :return: A BaseCMF object.
        """
        data['cmf'] = CMF(
            matrices={sp.sympify(k): Matrix(sp.sympify(v)) for k, v in data['cmf'].items()},
            validate=False
        )
        data['shifts'] = cls._shift_from_json(data['shifts'])
        data['selected_start_points'] = cls._selected_start_points_from_json(data['selected_start_points'])
        return cls(**data)

    def _to_json_obj(self) -> dict:
        """
        Converts the pFq to a JSON string (i.e., convert sp.Expr to str)
        :return: A dictionary representation of the pFq matching the JSON format.
        """
        return {
            **super()._to_json_obj(),
            **{sp.srepr(sym): sp.srepr(matrix) for sym, matrix in self.cmf.matrices.items()},
        }

    def to_cmf(self) -> ShiftCMF:
        """
        Converts the CMF to a Shift CMF.
        :return: A Shift CMF object
        """
        self.shifts = Position({k: v for k, v in zip(self.cmf.matrices.keys(), self.shifts)})
        return ShiftCMF(self.cmf, self.shifts, self.selected_start_points, self.only_selected, self.use_inv_t)

    def __repr__(self):
        return json.dumps(self._to_json_obj())

    def __str__(self):
        return f'<{self.__class__.__name__}: {self.__repr__()}>'

    def __hash__(self):
        return hash((hash(self.cmf), super().__hash__()))

