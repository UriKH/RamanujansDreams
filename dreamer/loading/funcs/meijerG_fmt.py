import json
from ramanujantools.cmf.meijer_g import MeijerG as rt_mg
from dreamer.utils.constants.constant import Constant
from dreamer.loading.funcs.formatter import Formatter
from dreamer.utils.types import *
from dreamer.configs import config


class MeijerG(Formatter):
    def __init__(self,
                 const: str | Constant, m: int, n: int, p: int, q: int, z: sp.Expr | int, shifts: Optional[list] = None,
                 selected_start_points: Optional[List[Tuple[Union[int, sp.Rational], ...]]] = None,
                 only_selected: bool = False,
                 use_inv_t: bool = config.search.DEFAULT_USES_INV_T
                 ):
        """
        Represents a pFq and its CMF + allows conversion to and from JSON.
        :var const: The constant related to this pFq function
        :var m: The m value of Meijer G.
        :var n: The n value of Meijer G.
        :var p: The p value of Meijer G.
        :var q: The q value of Meijer G.
        :var z: The z value of Meijer G.
        :var shifts: The shifts in starting point in the CMF where a sp.Rational indicates a shift.
        While 0 indicates no shift (None if not doesn't matter).
        :param selected_start_points: Optional list of start points to extract shards from.
        :param only_selected: If True, only extract shards from the selected start points.
        """
        super().__init__(const, use_inv_t)
        self.m = m
        self.n = n
        self.p = p
        self.q = q
        self.z = z
        self.shifts = shifts
        if self.shifts is None:
            self.shifts = [0] * (self.p + self.q)

        if self.p <= 0 or self.q <= 0:
            raise ValueError("Non-positive values")
        if not isinstance(self.shifts, list) and not isinstance(self.shifts, Position):
            raise ValueError("Shifts should be a list or Position")
        if self.p + self.q != len(self.shifts) and len(self.shifts) != 0:
            raise ValueError("Shifts should be of length p + q or 0")
        self.selected_start_points = selected_start_points
        self.only_selected = only_selected
        self.use_inv_t = use_inv_t

    @classmethod
    def _from_json_obj(cls, data: dict | list) -> "MeijerG":
        """
        Converts a JSON string to a MeijerG.
        :param s_json: The JSON string representation of the MeijerG (only attributes).
        :return: A MeijerG object.
        """
        data['z'] = sp.sympify(data['z']) if isinstance(data['z'], str) else data['z']
        data['shifts'] = [sp.sympify(shift) if isinstance(shift, str) else shift for shift in data['shifts']]
        return cls(**data)

    def _to_json_obj(self) -> dict:
        """
        Converts the MeijerG to a JSON string (i.e., convert sp.Expr to str)
        :return: A dictionary representation of the MeijerG matching the JSON format.
        """
        return {
            **super()._to_json_obj(),
            "m": self.m,
            "n": self.n,
            "p": self.p,
            "q": self.q,
            "z": str(self.z) if isinstance(self.z, sp.Expr) else self.z,
            "shifts": [str(shift) if isinstance(shift, sp.Expr) else shift for shift in self.shifts]
        }

    def to_cmf(self) -> ShiftCMF:
        """
        Converts the MeijerG to a CMF.
        :return: A tuple (CMF, shifts)
        """
        cmf = rt_mg(self.m, self.n, self.p, self.q, self.z)
        self.shifts = Position({k: v for k, v in zip(cmf.matrices.keys(), self.shifts)})
        return ShiftCMF(cmf, self.shifts, self.selected_start_points, self.only_selected, self.use_inv_t)

    def __repr__(self):
        return json.dumps(self._to_json_obj())

    def __str__(self):
        return f'<{self.__class__.__name__}: {self.__repr__()}>'

    def __hash__(self):
        return hash((self.m, self.n, self.p, self.q, self.z, self.shifts))

